# 在训练前添加读取已有训练迭代次数的逻辑
def get_total_iterations(training_loss_csv_path):
    # 读取训练损失 CSV 文件
    loss_data = try_read_csv(training_loss_csv_path, info="训练损失", header=None, isPrintInfo=False)
    # 获取CSV的行数
    if not loss_data.empty:
        # CSV文件中每一行是一次迭代的记录，我们根据行数来判断迭代次数
        return len(loss_data)
    return 0  # 如果没有找到CSV文件或文件为空，返回0

def Train_Model():
    num_epochs = 1000
    max_iterations = 50000  # 目标最大迭代次数

    # 获取当前已训练的迭代次数
    current_iterations = get_total_iterations(training_loss_csv_path)
    print(f"当前迭代次数: {current_iterations}")

    # 如果已经超过了最大迭代次数，直接退出训练
    if current_iterations >= max_iterations:
        print(f"已达到最大迭代次数 {max_iterations}, 训练结束.")
        return

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parameters are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    loss_info = []
    try:
        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for batch_idx, nbatch in enumerate(tepoch):
                        try:
                            # data normalized in dataset
                            # device transfer
                            nimage_1 = nbatch['image_1'][:,:obs_horizon].to(device)
                            nimage_2 = nbatch['image_2'][:,:obs_horizon].to(device)
                            naction = nbatch['action'].to(device)

                            # encoder vision features
                            image_features_1 = nets['vision_encoder_1'](
                                nimage_1.flatten(end_dim=1))
                            image_features_1 = image_features_1.reshape(
                                *nimage_1.shape[:2],-1)
                            B = image_features_1.shape[0]
                            image_features_2 = nets['vision_encoder_2'](
                                nimage_2.flatten(end_dim=1))
                            image_features_2 = image_features_2.reshape(
                                *nimage_2.shape[:2],-1)

                            # concatenate vision feature and low-dim obs
                            obs_features = torch.cat([image_features_1, image_features_2], dim=-1)
                            obs_cond = obs_features.flatten(start_dim=1)

                            # sample noise to add to actions
                            noise = torch.randn(naction.shape, device=device)

                            # sample a diffusion iteration for each data point
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (B,), device=device
                            ).long()

                            # add noise to the clean images according to the noise magnitude at each diffusion iteration
                            # (this is the forward diffusion process)
                            noisy_actions = noise_scheduler.add_noise(
                                naction, noise, timesteps)

                            # predict the noise residual
                            noise_pred = noise_pred_net(
                                noisy_actions, timesteps, global_cond=obs_cond)

                            # L2 loss
                            loss = nn.functional.mse_loss(noise_pred, noise)

                            # optimize
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            # step lr scheduler every batch
                            # this is different from standard pytorch behavior
                            lr_scheduler.step()

                            # update Exponential Moving Average of the model weights
                            ema.step(noise_pred_net)

                            # logging
                            loss_cpu = loss.item()
                            epoch_loss.append(loss_cpu)
                            tepoch.set_postfix(loss=loss_cpu)
                            
                            loss_info.append([epoch_idx + 1, batch_idx + 1, loss_cpu])
                            
                            # 每batch迭代保存模型
                            if (batch_idx + 1) % 50 == 0:
                                save_model(num_epochs, optimizer, lr_scheduler, ema)
                                # 将当前损失写入 CSV 文件 
                                try_to_csv(training_loss_csv_path, 
                                        pd.DataFrame(np.array(loss_info).reshape(-1,3)),
                                        info="训练损失", index=False, header=False, mode='a', isPrintInfo=True)
                                loss_info = []
                        
                        except Exception as e:
                            print(f"Batch {batch_idx} failed with error: {e}")
                            raise  # 重新抛出异常以触发外层的处理
                    

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                
                # 每epoch保存模型
                if (epoch_idx + 1) % 1 == 0:
                    save_model(num_epochs, optimizer, lr_scheduler, ema)
                    # 将当前损失写入 CSV 文件 
                    try_to_csv(training_loss_csv_path, 
                            pd.DataFrame(np.array(loss_info).reshape(-1,3)),
                            info="训练损失", index=False, header=False, mode='a', isPrintInfo=True)
                    loss_info = []

                # 每次epoch后检查当前迭代次数
                current_iterations = get_total_iterations(training_loss_csv_path)
                if current_iterations >= max_iterations:
                    print(f"已达到最大迭代次数 {max_iterations}, 训练结束.")
                    return  # 如果达到了最大迭代次数，则停止训练

        # Weights of the EMA model
        # is used for inference
        ema_noise_pred_net = noise_pred_net
        ema.copy_to(ema_noise_pred_net.parameters())
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # 重启脚本
        subprocess.Popen([sys.executable, *sys.argv])
        sys.exit(1)

