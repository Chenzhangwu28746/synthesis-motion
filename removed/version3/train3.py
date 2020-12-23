import datetime
import os
import torch
import pathlib
from removed.version4 import data_utils
from removed.version4.dataloader import MyDataset
#from removed.version2.tailorBvh import MyDataset
from removed.version4.model import EncoderTCN, AttributeTCN, Generator, Discriminator


            # =================================================================================== #
            #                             1. Initialize configurations.                           #
            # =================================================================================== #





"""设置训练结果保存路径"""
CurrentExperiment = "Experiment2"





"""上下文管理器,用于打开或关闭autograd引擎的异常检测."""
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

dataset = MyDataset()
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True, drop_last=True)

learning_rate = 1e-4
num_epochs = 100
num_classes = dataset.category_num


            # =================================================================================== #
            #                             1. Initialization model.                                #
            # =================================================================================== #

I = EncoderTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  latent_dim=64,
  category_num=num_classes,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.2
).to(device)

A = AttributeTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  latent_dim=64,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.2
).to(device)

G = Generator(
  id_dim=64,
  a_dim=64
).to(device)

C = EncoderTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  latent_dim=64,
  category_num=num_classes,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.2
).to(device)
# C.load_state_dict(torch.load(i_path)) # 直接使用预训练好的 I

D = Discriminator(
  input_size=241, # 骨骼1帧 + 动作240帧
  level_channel_num=256, # 提取器 [256]
  level_num=8,
  kernel_size=3,
  dropout=0.2
).to(device)



cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()

            # =================================================================================== #
            #                             1. Functional function                                  #
            # =================================================================================== #

def reparameterize(mu, logvar):
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return eps * std + mu

def save_models(dirpath):
  pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
  torch.save(I.state_dict(), os.path.join(dirpath, 'I.pt'))
  torch.save(C.state_dict(), os.path.join(dirpath, 'C.pt'))
  torch.save(A.state_dict(), os.path.join(dirpath, 'A.pt'))
  torch.save(D.state_dict(), os.path.join(dirpath, 'D.pt'))
  torch.save(G.state_dict(), os.path.join(dirpath, 'G.pt'))

def load_models(dirpath):
  I.load_state_dict(torch.load(os.path.join(dirpath, 'I.pt')))
  C.load_state_dict(torch.load(os.path.join(dirpath, 'C.pt')))
  A.load_state_dict(torch.load(os.path.join(dirpath, 'A.pt')))
  D.load_state_dict(torch.load(os.path.join(dirpath, 'D.pt')))
  G.load_state_dict(torch.load(os.path.join(dirpath, 'G.pt')))

# def save_models(path, resume_iters):
#   torch.save(I.state_dict(), os.path.join(path, '{}-I.pt'.format(resume_iters)))
#   torch.save(C.state_dict(), os.path.join(path, '{}-C.pt'.format(resume_iters)))
#   torch.save(A.state_dict(), os.path.join(path, '{}-A.pt'.format(resume_iters)))
#   torch.save(D.state_dict(), os.path.join(path, '{}-D.pt'.format(resume_iters)))
#   torch.save(G.state_dict(), os.path.join(path, '{}-G.pt'.format(resume_iters)))
#   print('Saved model checkpoints into {}...'.format(path))


# def restore_model(model_save_dir, resume_iters):
#   """Restore the trained generator and discriminator."""
#   print('Loading the trained models from step {}...'.format(resume_iters))
#   G_path = os.path.join(model_save_dir, '{}-G.pt'.format(resume_iters))
#   D_path = os.path.join(model_save_dir, '{}-D.pt'.format(resume_iters))
#   A_path = os.path.join(model_save_dir, '{}-A.pt'.format(resume_iters))
#   I_path = os.path.join(model_save_dir, '{}-I.pt'.format(resume_iters))
#   C_path = os.path.join(model_save_dir, '{}-C.pt'.format(resume_iters))
#   A.load_state_dict(torch.load(A_path, map_location=lambda storage, loc: storage))
#   I.load_state_dict(torch.load(I_path, map_location=lambda storage, loc: storage))
#   C.load_state_dict(torch.load(C_path, map_location=lambda storage, loc: storage))
#   G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
#   D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

#restore_model('./models/9-18-41','9-18-41')



#load_models('./models/轮次40')
i_optimizer = torch.optim.Adam(I.parameters(), learning_rate)
a_optimizer = torch.optim.Adam(A.parameters(), learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), learning_rate)
c_optimizer = torch.optim.Adam(C.parameters(), learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), learning_rate)

# 联合训练

            # =================================================================================== #
            #                             1. training                                             #
            # =================================================================================== #


for epoch in range(num_epochs):
  for batch_idx, (data_a, labels_a, data_b, labels_b) in enumerate(dataloader):

    batch = data_a.size(0)

    x_s = data_a.to(device=device)
    x_a = data_b.to(device=device)
    labels_a = labels_a.to(device=device)
    # labels_b = labels_b.to(device=device) # 暂时用不到




    print('# 当前轮次：{}，当前批次：{}'.format(epoch, batch_idx))
    ## 自交杂交轮流进行
    if (batch_idx % 2 == 1):
      # 奇数步自交
      lbd = 1  # 参数 λ
      x_a = x_s
      print('## 自交，λ：{}'.format(lbd))
    else:
      # 偶数步杂交
      lbd = 0.1
      print('## 杂交，λ：{}'.format(lbd))

    ################## 训练 I、C、A、D

    i_ctg, i_id = I(x_s) # ctg 代表 category
    c_ctg, _ = C(x_s)

    loss_I = cross_entropy_loss(i_ctg, labels_a)
    loss_C = cross_entropy_loss(c_ctg, labels_a)
    print('loss_I', loss_I.item())
    print('loss_C', loss_C.item())

    a_mu, a_log_var = A(x_a) # VAE 的均值和标准差
    a_z = reparameterize(a_mu, a_log_var) # 从均值和标准差组成的正态分布里采出一个 z
    loss_KL = torch.mean(0.5 * (torch.pow(a_mu, 2) + torch.exp(a_log_var) - a_log_var - 1)) #  贴近正态分布的约束
    print('loss_KL', loss_KL.item())

    x_f = G(i_id, a_z) # f 代表 fake

    d_real_p, d_real_feature = D(x_s)
    d_fake_p, d_fake_feature = D(x_f)
    # loss_D = torch.mean(-torch.log(d_real_p) - torch.log(1 - d_fake_p)) # 会出 log0 nan
    loss_D = (bce_loss(d_real_p, torch.ones_like(d_real_p)) + bce_loss(d_fake_p, torch.zeros_like(d_fake_p))) / 2

    print('loss_D', loss_D.item())

    i_optimizer.zero_grad()
    c_optimizer.zero_grad()
    d_optimizer.zero_grad()
    a_optimizer.zero_grad()

    loss_1 = loss_I + loss_C + loss_KL + loss_D
    loss_1.backward() #retain_graph=True
    i_optimizer.step()
    c_optimizer.step()
    d_optimizer.step()
    a_optimizer.step()

    ################## 训练 G
    _, i_id = I(x_s)
    a_mu,a_log_var = A(x_a)
    a_z = reparameterize(a_mu,a_log_var)
    x_f = G(i_id, a_z) # f 代表 fake

    d_real_p, d_real_feature = D(x_s)
    d_fake_p, d_fake_feature = D(x_f)
    # loss_GD = torch.mean(torch.diagonal(torch.cdist(d_fake_feature, d_real_feature)) * (1 / 2)) # 这里计算了n²个距离，然后抓出对角线，有性能优化空间
    # loss_GR = torch.mean(torch.diagonal(torch.cdist(x_f.reshape(batch, -1), x_a.view(batch, -1))) * (lbd / 2)) # 把 x_a 展平，计算 x_a 与 x_f 平方差距离
    loss_GD = (1/2) * mse_loss(d_fake_feature, d_real_feature)
    loss_GR = (lbd/2) * mse_loss(x_f.reshape(batch, -1), x_a.view(batch, -1))

    print('loss_GD', loss_GD.item())
    print('loss_GR', loss_GR.item())

    _, c_fake_id = C(x_f)
    # loss_GC = torch.mean(torch.diagonal(torch.cdist(c_fake_id, i_id)) * (1 / 2))
    loss_GC = (1/2) * mse_loss(c_fake_id, i_id)
    print('loss_GC', loss_GC.item())

    g_optimizer.zero_grad()
    loss_2 = lbd * loss_GR + loss_GD + loss_GC
    loss_2.backward()
    g_optimizer.step()

    date = str(datetime.datetime.now().day) + '-' + str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute)

    if epoch % 5 == 0 and (batch_idx % 5000 == 0 or batch_idx % 5000 == 1):
      # statistics = np.loadtext('./v5/walk_id_compacted/_min_max_mean_std.csv')
      data_utils.save_bvh_to_file(
        './outputs/'+CurrentExperiment+'/轮次{}-批次{}-{}.bvh'.format(epoch, batch_idx, '自交' if batch_idx % 2 == 1 else '杂交'),
        data_utils.denormalized(x_f[0].cpu().detach(), dataset.statistics)
      )
    if epoch % 10 == 0 and batch_idx == 0:
      save_models('./models/'+CurrentExperiment+'/轮次{}'.format(epoch))

"""以当前时间命名预训练模型"""

# os.mkdir('./models/'+string)
# save_models('./models/'+string, string)


    ## 上面是原始代码的训练步骤
    ## 下面是论文中的训练步骤
    ## 都会遇到原地操作 Good Luck 报错

    # i_optimizer.zero_grad()
    # loss_I.backward(retain_graph=True)
    # i_optimizer.step()

    # c_optimizer.zero_grad()
    # loss_C.backward(retain_graph=True)
    # c_optimizer.step()

    # d_optimizer.zero_grad()
    # loss_D.backward(retain_graph=True)
    # d_optimizer.step()

    # g_optimizer.zero_grad()
    # loss_G = lbd * loss_GR + loss_GD + loss_GC
    # loss_G.backward(retain_graph=True)
    # g_optimizer.step()

    # loss_A = lbd * (loss_KL + loss_G)
    # loss_A.backward()
    # a_optimizer.step()

