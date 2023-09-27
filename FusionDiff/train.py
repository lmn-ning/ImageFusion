import time
import torch
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset import MFI_Dataset
from Diffusion import GaussianDiffusion
from Condition_Noise_Predictor.UNet import NoisePred
from utils import tensorboard_writer, logger, save_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config_path):
    timestr = time.strftime('%Y%m%d_%H%M%S')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # train dataset
    train_datasePath = config["dataset"]["train"]["path"]
    train_phase = config["dataset"]["train"]["phase"]
    train_batch_size = config["dataset"]["train"]["batch_size"]
    train_use_dataTransform = config["dataset"]["train"]["use_dataTransform"]
    train_resize = config["dataset"]["train"]["resize"]
    train_imgSize = config["dataset"]["train"]["imgSize"]
    train_shuffle = config["dataset"]["train"]["shuffle"]
    train_drop_last = config["dataset"]["train"]["drop_last"]
    train_dataset = MFI_Dataset(train_datasePath, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                resize=train_resize, imgSzie=train_imgSize)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=train_drop_last)

    # Condition Noise Predictor
    in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
    out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
    model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
    num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
    dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
    time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
    down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
    model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                      down_sample_mult)

    # whether to use the pre-training model
    use_preTrain_model = config["Condition_Noise_Predictor"]["use_preTrain_model"]
    if use_preTrain_model:
        preTrain_Model_path = config["Condition_Noise_Predictor"]["preTrain_Model_path"]
        model.load_state_dict(torch.load(preTrain_Model_path, map_location=device))
        print(f"using pre-trained model：{preTrain_Model_path}")
    model = model.to(device)

    # channel splicing mode
    concat_type = config["Condition_Noise_Predictor"]["concat_type"]
    assert concat_type in ["ABX", "AXB", "XAB"], "Check that the 'concat_type' parameter is correct"

    # optimizer
    init_lr = config["optimizer"]["init_lr"]
    use_lr_scheduler = config["optimizer"]["use_lr_scheduler"]
    StepLR_size = config["optimizer"]["StepLR_size"]
    StepLR_gamma = config["optimizer"]["StepLR_gamma"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    if use_lr_scheduler:
        learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size=StepLR_size, gamma=StepLR_gamma)

    # diffusion model
    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    loss_scale = config["diffusion_model"]["loss_scale"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)

    # log
    writer = tensorboard_writer(timestr)
    log = logger(timestr)
    print(f"time: {timestr}")
    log.write(f"time: {timestr} \n")
    print(f"using {len(train_dataset)} images for train")
    log.write(f"using {len(train_dataset)} images for train  \n\n")
    log.write(f"config:  \n")
    log.write(json.dumps(config, ensure_ascii=False, indent=4))
    if use_lr_scheduler:
        log.write(
            f"\n learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size={StepLR_size}, gamma={StepLR_gamma})  \n\n")

    # hyper-parameter
    epochs = config["hyperParameter"]["epochs"]
    start_epoch = config["hyperParameter"]["start_epoch"]
    loss_step = config["hyperParameter"]["loss_step"]
    save_model_epoch_step = config["hyperParameter"]["save_model_epoch_step"]
    train_step_sum = len(train_dataloader)
    num_train_step = 0

    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        loss_sum = 0
        writer.add_scalar('lr_epoch: ', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        for train_step, train_images in tqdm(enumerate(train_dataloader), desc="train step"):
            optimizer.zero_grad()
            train_sourceImg1 = train_images[0].to(device)
            train_sourceImg2 = train_images[1].to(device)
            clearImg = train_images[2].to(device)

            t = torch.randint(0, T, (train_batch_size,), device=device).long()
            scale_loss = diffusion.train_losses(model, train_sourceImg1, train_sourceImg2, clearImg, t, concat_type, loss_scale)
            writer.add_scalar('loss_step: ', scale_loss, num_train_step)

            if train_step % loss_step == 0:
                print(
                    f" [epoch] {epoch}/{epochs}    "
                    f"[epoch_step] {train_step}/{train_step_sum}     "
                    f"[train_step] {num_train_step}     "
                    f"[loss] {scale_loss.item() / loss_scale :.6f}     "
                    f"[scale_loss] {scale_loss.item() :.6f}     "
                    f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                    f"[t] {t.cpu().numpy()}")

                log.write(f" [epoch] {epoch}/{epochs}    "
                          f"[epoch_step] {train_step}/{train_step_sum}     "
                          f"[train_step] {num_train_step}     "
                          f"[loss] {scale_loss.item() / loss_scale :.6f}     "
                          f"[scale_loss] {scale_loss.item() :.6f}     "
                          f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                          f"[t] {t.cpu().numpy()}"
                          f"\n")

            scale_loss.backward()
            optimizer.step()

            loss_sum += scale_loss
            num_train_step += 1

        aver_loss = loss_sum / train_step_sum

        if epoch % save_model_epoch_step == 0:
            save_model(model, epoch, timestr)
        if epoch == epochs - 1:
            save_model(model, epoch, timestr)

        # update learning rate
        if use_lr_scheduler:
            learningRate_scheduler.step()
        writer.add_scalar('aver_loss_epoch: ', aver_loss, epoch)
        log.write("\n")

    print("End of training")
    log.write("End of training \n")
    writer.close()


if __name__ == '__main__':
    config_path = "config.json"
    train(config_path)
