import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from data import create_dataset
import torch
import torchvision

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    phase = 'train'
    train_dataset = create_dataset(phase)  # create a dataset given opt.dataset_mode and other options
    train_dataloader = train_dataset.dataloader
    print('The number of training images = %d' % len(train_dataset))
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.

    phase = 'test'
    val_dataset = create_dataset(phase,'Mayo2020')
    val_dataloader = val_dataset.dataloader
    print('The number of training images = %d' % len(val_dataset))
    val_dataset_size = len(val_dataset)

    batch_size = opt.batch_size
    opt.model = 'cycle_gan'
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        running_psnr = 0
        running_ssim = 0

        for i, data in enumerate(train_dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            psnr, ssim, y_pred, y,path = model.compute_metrics()

            running_psnr += psnr
            running_ssim += ssim

        epoch_psnr = running_psnr / len(train_dataloader)
        epoch_ssim = running_ssim / len(train_dataloader)

        print('Epoch: [{}/{}], train_psnr: {:.4f}, train_ssim: {:.4f}'.format(epoch, opt.n_epochs, epoch_psnr, epoch_ssim))

        if epoch % 20==0:
            test_running_psnr = 0
            test_running_ssim = 0

            with torch.no_grad():
                model.eval()
                for i, data in enumerate(val_dataloader):
                    model.set_input(data)  # unpack data from data loader
                    model.test()  # run inference

                    psnr, ssim, y_pred, y,path = model.compute_metrics()
                    test_running_psnr += psnr
                    test_running_ssim += ssim

                    
                epoch_test_psnr = test_running_psnr / len(val_dataloader)
                epoch_test_ssim = test_running_ssim / len(val_dataloader)

            print('val:Epoch: [{}/{}], test_psnr: {:.4f}, test_ssim: {:.4f}'.format(epoch, opt.n_epochs, epoch_test_psnr,
                                                                                    epoch_test_ssim))

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('cycle')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

