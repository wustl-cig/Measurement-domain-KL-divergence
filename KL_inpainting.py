from PIL import Image
import click
from tqdm import tqdm
import pickle
from torchvision import transforms as T
from skimage.metrics import  mean_squared_error

from EDM.util.utils import *
from EDM import dnnlib
import torch_utils
toTensor = T.ToTensor()


@click.command()
@click.option('--outdir',                  help='Where to save the outputs',                    default='./results/KL/',                     type=str)
@click.option('--dataset_path',            help='path to testset',                              default='./testsets/FFHQ/FFHQ_20',          type=str)
@click.option('--gpu-ids',                 help='which GPU to use',                             default = 2,                                type=int)
@click.option('--img_size',                help='size of the test images',                      default = 64,                               type = int)
@click.option('--prob',                    help='inpainting mask probablity',                   default= 0.5,                               type=float)
@click.option('--patch',                   help='inpainting patch size',                        default = 16,                               type = int)
@click.option('--ifnoise',                 help= 'if measurements are noisy',                   default = False,                            type = bool)
@click.option('--meas_noise',              help= 'measuremenet noise level',                    default = 0.0 ,                             type = float)
@click.option('--net_list',                help= 'List of OOD models',                          default = "ffhq",                               type = str)

def main(**sampler_kwargs):

    torch.manual_seed(42)
    opts = dnnlib.EasyDict(sampler_kwargs)
    device_str = f"cuda:{opts.gpu_ids}" if torch.cuda.is_available() else 'cpu'
    print(f"Device set to {device_str}.")
    device = torch.device(device_str)


    # Loading models (InD and OOD)
    models_list = parse_str_list(opts.net_list)
    rootpath_models = './model_zoo/inpainting'
    models_list = get_net_list(rootpath_models, models_list)
    net = {}
    print('Loading networks from network lists ...')
    for network in models_list:
        print(f'Loading network from "{network}"...')
        with open(network, 'rb') as file:
            net[network] = pickle.load(file)['ema'].to(device)

    with open(os.path.join(rootpath_models, 'ffhq.pkl'), 'rb') as file:
        net['matched'] = pickle.load(file)['ema'].to(device)

    # Loading images from testset
    if os.path.isdir(opts.dataset_path):
        paths = sorted([os.path.join(opts.dataset_path, x) for x in os.listdir(opts.dataset_path)])
    else:
        paths = [opts.dataset_path]
    all_images = []
    for i, path in enumerate(paths):
        orig_im = Image.open(path)
        all_images.append(scale(opts.img_size, opts.img_size, orig_im))


    mse_records, mse_records_den, metric_den, metrics , metric_den['mse'], mean_MSE, mean_MSE_noise = {}, {},  {},  {}, {}, {}, {}
    sigma_range, metrics['time_range'], metrics['sigma_range'], metrics['mse']  = [], [], [], []

    for time in range(0, 40, 1):
        mse_records[time] = {}
        mse_records_den[time] = {}
        for network_key in models_list:
            mse_records[time][network_key] = []
            mse_records_den[time][network_key]  = []


    for time in tqdm(range(0, 40, 1)):
        for i in all_images:

            im = normalize(toTensor(i)).unsqueeze(0).to(device)
            y_noise = im
            b, c, h, w = im.shape

            # Prepare inpainting mask
            masks = (torch.rand((b, 1, opts.patch, opts.patch), device=device) > opts.prob).int()
            masks = masks.repeat(1, c, 1, 1)
            masks = masks.repeat_interleave(h // opts.patch, 2).repeat_interleave(w // opts.patch, 3)
            y_inpainting = im * masks + 0.5*(1-masks)

            # Add noise to measurements if ifnoise is True
            if opts.ifnoise:
                y_inpainting =  y_inpainting + opts.meas_noise * torch.randn_like(im)

            # Feed to InD diffusion model
            den_img_matched_denoised, sigma, noisy = denoise_net(net['matched'], y_noise, time=time, device=device) # Img. domain
            den_img_matched_inpaint, sigma, noisy = denoise_net(net['matched'], y_inpainting, time=time,device=device) # Meas. domain

            for network_key in models_list:
                # Feed to OOD diffusion models
                den_img_denoising, sigma, noisy = denoise_net(net[network_key], y_noise, time = time, device = device)
                den_img_inpainting, sigma, noisy = denoise_net(net[network_key], y_inpainting, time=time, device=device)

                # Score function gap calculation
                mse_den = mean_squared_error(clear(den_img_matched_denoised), clear(den_img_denoising))
                mse_inpaint =  mean_squared_error(clear(den_img_matched_inpaint), clear(den_img_inpainting))

                mse_records[time][network_key].append(mse_inpaint)
                mse_records_den[time][network_key].append(mse_den)

        sigma_range.append(sigma)


    os.makedirs(opts.outdir, exist_ok=True)
    if opts. ifnoise:
        name_folder = opts.outdir + 'prob_'+str(opts.prob) +'nois_'+str(opts.meas_noise)+'n_samples_'+ opts.dataset_path.split('/')[-1]+ '.pkl'
    else:
        name_folder = opts.outdir + 'prob_'+str(opts.prob) +'n_samples_'+ opts.dataset_path.split('/')[-1]+ '.pkl'


    metrics = {'mse_inpaint':mse_records, 'mse_den':mse_records_den,  'sigma':sigma_range}
    with open(name_folder, 'wb') as pickle_file:
        pickle.dump(metrics, pickle_file)


    print("successfully computed KL divergence")


if __name__ == "__main__":
    main()
