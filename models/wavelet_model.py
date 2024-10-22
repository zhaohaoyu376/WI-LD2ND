import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.util as util
import torch.nn.functional as F


class waveletmodel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--netF', type=str, default='mlp_sample',
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--netF', type=str, default='batch',
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--no_dropout', type=str, default='batch',
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--netF_nc', type=int, default=256)
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # =====================================================================
        self.m = 0.99
        self.num_patches = [256]
        self.nce_layers = [0, 4, 8, 12, 16]
        input_nc = 1
        netF = 'mlp_sample'
        normG = 'batch'
        netF_nc = 256
        # self.criterionNCE = []
        netOnline = networks.Disentangle_UNet(n_channels=1, n_classes=1, bilinear=True)
        self.netOnline = networks.init_net(netOnline, gpu_ids=opt.gpu_ids, )
        netTarget = networks.Disentangle_UNet(n_channels=1, n_classes=1, bilinear=True)
        self.netTarget = networks.init_net(netTarget, gpu_ids=opt.gpu_ids, )
        self.netProjection_online = networks.define_F(input_nc=input_nc, netF=netF, norm=normG,
                                                      gpu_ids=self.gpu_ids, nc=netF_nc)
        self.netProjection_target = networks.define_F(input_nc=opt.input_nc, netF=netF, norm=normG,
                                                      gpu_ids=self.gpu_ids, nc=netF_nc)

        self.optimizer_F = torch.optim.SGD(
            list(self.netOnline.parameters()) + list(self.netProjection_online.parameters()), lr=self.opt.lr)
        self.dwt = networks.DWT_transform(1, 1)
        self.dwt.to(self.gpu_ids[0])
        self.dwt = torch.nn.DataParallel(self.dwt, self.gpu_ids)
        # for nce_layer in self.nce_layers:
        #     self.criterionNCE.append(networks.PatchNCELoss(opt).to(self.device))
        # =====================================================================

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # if self.isTrain:  # define discriminators
        #     self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
        #                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        #     self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
        #                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # self.criterionCycle = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        self.loss_G = torch.nn.MSELoss()(self.fake_B, self.real_B) + (1 - util.compute_ssim(self.fake_B, self.real_B))

        # feat_real_low, feat_real_high = self.dwt(self.real_B)
        # feat_fake_low, feat_fake_high = self.dwt(self.fake_B)

        # gradient_loss = self.gradient_loss(feat_real_high,feat_fake_high)
        # self.loss_G += 0.01*gradient_loss

        self.loss_F = self.backward_F()
        #self.loss_G += 0.01 * self.loss_F
        self.loss_G.backward()

    def backward_F(self):

        torch.set_grad_enabled(True)


        feat_real_low, feat_real_high = self.dwt(self.real_B)
        feat_fake_low, feat_fake_high = self.dwt(self.fake_B)
        

        patch_size = feat_real_low.shape[-1]


        feat_real_high = self.netTarget(feat_real_high)
        feat_pool_real_high, sample_ids, sample_local_ids, sample_top_idxs = self.netProjection_target(patch_size,
                                                                                                  feat_real_high,
                                                                                                  self.num_patches, )

        feat_fake_high = self.netOnline(feat_fake_high)
        feat_pool_fake_high, _, _, _ = self.netProjection_online(patch_size, feat_fake_high,
                                                            self.num_patches, sample_ids,
                                                            sample_local_ids, sample_top_idxs)

        total_nce_loss = 0.0
        weight = 1

        for i,(f_q_1, f_k_1) in enumerate(zip(feat_pool_real_high,feat_pool_fake_high)):
        #     # if i==2 or i==1:
            if i==1:
                loss = self.regression_loss(f_q_1, f_k_1.detach())
                total_nce_loss += weight*loss.mean()
            if i==0:
                loss = networks.PatchNCELoss()(f_q_1, f_k_1.detach())
            #total_nce_loss += weight*loss.mean()


        return total_nce_loss

    def eval(self):
        self.netG_A.eval()
        # self.netG_B.eval()
        # self.netD_A.eval()
        # self.netD_B.eval()

    def gradient_loss(self, x, y):
        reconstruction_loss = F.mse_loss(x, y)

        gradient_x = F.conv2d(x, weight=torch.Tensor([[-1, 0, 1]]).cuda().view(1, 1, 1, 3), padding=(0, 1), stride=1)
        gradient_y = F.conv2d(x, weight=torch.Tensor([[-1], [0], [1]]).cuda().view(1, 1, 3, 1), padding=(1, 0),
                              stride=1)
        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        # Calculate weighted reconstruction loss based on gradients
        gradient_loss = torch.mean(10 * gradient_magnitude * reconstruction_loss) + reconstruction_loss

        return gradient_loss

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def optimize_parameters(self):

        # self.loss_F = self.backward_F(self.real_B,self.fake_B)

        self.forward()  # compute fake images and reconstruction images.

        # =================================================
        # =================================================
        #self.set_requires_grad(self.netOnline, True)
        #self.set_requires_grad(self.netProjection_online, True)
        #self.optimizer_F.zero_grad()

        #self.loss_F = self.backward_F()
        #self.loss_F.backward()
        #self.optimizer_F.step()
        #self._update_target_network_parameters()

        #self.set_requires_grad(self.netOnline, False)
        #self.set_requires_grad(self.netProjection_online, False)
        # =================================================
        # =================================================

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

    def compute_metrics(self):
        with torch.no_grad():
            y_pred = self.fake_B
            y = self.real_B
            x = self.real_A
            path = self.image_paths

            psnr = util.compute_psnr(y_pred, y)
            ssim = util.compute_ssim(y_pred, y)

            return psnr, ssim, y_pred, y, path

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.netOnline.parameters(), self.netTarget.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


        for param_q, param_k in zip(self.netProjection_online.parameters(), self.netProjection_target.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
