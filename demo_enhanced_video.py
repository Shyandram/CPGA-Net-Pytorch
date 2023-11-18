import os
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm
from utils import weight_init #,logger
from config import get_config
from model import enhance_color
from data import LLIEDataset
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from skimage import img_as_ubyte
import cv2, time
from PIL import Image

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

def load_network(device):
    net = enhance_color().to(device)
    net.apply(weight_init)
    return net


def load_pretrain_network(cfg, device):
    net = enhance_color().to(device)
    net.load_state_dict(torch.load(os.path.join(cfg.ckpt))['state_dict'])
    return net


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # # load summaries
    # summary = load_summaries(cfg)
    # -------------------------------------------------------------------
    # load data
    w, h = 640, 480

    data_transform = transforms.Compose([
        transforms.Resize([ h, w]),
        # transforms.RandomCrop([256, 256]),
        # transforms.CenterCrop([400, 600]),
        transforms.ToTensor()
    ])

    if os.path.isdir(cfg.video_dir):
        val_haze_dataset = LLIEDataset(cfg.video_dir, cfg.video_dir, data_transform, dataset_type ='LOL-v1', isdemo=True)
        val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=1, shuffle=False,
                                       num_workers=0, drop_last=True, pin_memory=True)
    else:
        cap = cv2.VideoCapture(cfg.video_dir)
        if (cap.isOpened()== False): 
            raise Warning("Error opening video file")
    
    if not os.path.isdir(cfg.sample_output_folder):
        os.makedirs(cfg.sample_output_folder)
    # -------------------------------------------------------------------
    #
    # -------------------------------------------------------------------
    # load network
    global network
    if cfg.ckpt:
        network = load_pretrain_network(cfg, device)
    else:
        network = load_network(cfg.upscale_factor, device)
    
    # -------------------------------------------------------------------
    out_dir = os.path.join('results', cfg.net_name)
    os.makedirs(out_dir, exist_ok=True)

    # start train
    print('Start demo')
    network.eval()
    global start_time,c , gamma, gamma_lst
    start_time =  time.time_ns()
    c = 0
    prev_frame_time =  time.time_ns()*1e-9 
    
    gamma = 1
    gamma_lst = []
    #
    global videoWriter, videoWriter2
    output_name = cfg.output_name
    if not output_name:
        raise Warning("please provide a output name")
    videoWriter = cv2.VideoWriter(output_name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w,h))
    videoWriter2 = cv2.VideoWriter(output_name+'_display.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w+w,h))
    
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    global total
    total = 0
    
    if os.path.isdir(cfg.video_dir):   
        loader = tqdm(val_loader)
        for step, (ori_image, haze_image, _, name) in enumerate(loader):
            haze_image = haze_image.to(device)
            display(haze_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
    
    else:
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        # Read until video is completed 
            while(cap.isOpened()): 
                ret, frame = cap.read() 
                if ret == True: 
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame[:frame.shape[0]//2,:, :] = 0
                    haze_image = data_transform(Image.fromarray(frame)).unsqueeze(0)
                    haze_image = haze_image.to(device)
                    
                    display(haze_image)
                        
                    pbar.update(1)
            
                    # Press Q on keyboard to exit 
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        break
                else:
                    break
            
    print("total FPS: ", total/c)
    plt.plot(gamma_lst)
    plt.savefig(output_name+"gamma_v.png")
    
    videoWriter.release()
    videoWriter2.release()
    print('finish demo')
def display(haze_image):
    global gamma, c, total
    start_t = time.time_ns()
    with torch.no_grad():
            LLIE_image, gamma, t = network(haze_image, gamma, isvid=True)
            # if c%4 ==0:
            #     LLIE_image, gamma, t = network(haze_image, isvid=True)
            # else:
            #     LLIE_image, gamma, t = network.forward_ng(haze_image, gamma, t)
            LLIE_image = torch.clamp(LLIE_image, 0, 1)
            out = torch.cat((haze_image, LLIE_image), 3)
            
            out = out.permute(0, 2, 3, 1).cpu().detach().numpy()
            out = img_as_ubyte(out[0])
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
           
            videoWriter.write(cv2.cvtColor(img_as_ubyte(LLIE_image.permute(0, 2, 3, 1).cpu().detach().numpy()[0]), cv2.COLOR_RGB2BGR))
                        
            c+=1
            fps = 1/(time.time_ns()-start_t)*1e9
            total += fps
            
            gamma_lst.append(gamma.clone().detach().view(-1).cpu().numpy())
                        
            # putting the FPS count on the frame 
            cv2.putText(out, 'FPS:'+'%.2f'%(fps) +' ;Inf spd='+'%.2f'%(1/fps) + '; Gamma=%.2f'%(gamma), (7, 70), cv2.FONT_HERSHEY_SIMPLEX  
                        , 1, (255, 0, 255), 3, cv2.LINE_AA) 
            videoWriter2.write(out)
            cv2.imshow('Frame', cv2.resize(out, (out.shape[1], out.shape[0]))) 
if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
