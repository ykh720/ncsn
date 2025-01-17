from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import cm

class Dset(Dataset):

    def __init__( self , x_train , transform=True):
        
        self.transform = transform
        self.x=x_train
        if self.transform:
            self.x = torch.Tensor(self.x)
            # torch.tensor infers the dtype automatically, while torch.Tensor returns a torch.FloatTensor
            # torch.Tensor gives torch.float32
            # https://stackoverflow.com/questions/51911749/what-is-the-difference-between-torch-tensor-and-torch-tensor

    def __getitem__(self , index):
         return self.x[index]
        

    def __len__(self):
        return len(self.x)

def IVS_visualize(gen_row, Klist, tlist, savepath,   plotname = "",):
    """accept inputs from ANN and GAN"""
    if isinstance(gen_row, (np.ndarray, np.generic) ):
        testGANy = gen_row
    else:
        testGANy = gen_row.cpu().numpy()

    testGANy = testGANy.reshape((len(tlist), len(Klist)))

    X,Y = np.meshgrid(Klist, tlist)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    Z = testGANy
    surf = ax.plot_surface(X, Y, Z)

    ax.set_xlabel('Strike')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied volatility')

    #### 04-27
    # Invert the x-axis
    ax.set_xlim(ax.get_xlim()[::-1])

    # ax.set_title(plotname + ' IVS')

    ########### 0504    
    ax.set_title(plotname)

    plt.savefig(savepath)
    plt.close()
    # plt.show()

    # return Z 

##### OLD version
# def inpainting_error(surfivs, surfivspred, Klist, tlist, savepath, mask, ymlpath):
#     """surfivs shape (batch_size, image_size_y, image_size_x)"""

#     def rotate_xticklabels(ax, angle):
#         for tick in ax.get_xticklabels():
#             tick.set_rotation(angle)

#     from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
#     import matplotlib.pyplot as plt
#     import matplotlib.ticker as mtick
#     import torch
#     import numpy as np 

#     indices = np.nonzero(~mask)
#     indices = indices.tolist()
#     indices = [tuple(x) for x in indices]
#     # avgerror = [] 
#     # stderror = []

#     plt.figure(1,figsize=(14,4))
#     ax=plt.subplot(1,3,1)
#     # err = np.mean(100 * np.abs((surfivspred - surfivs)/ surfivs), axis =0) 
#     # print(surfivs.shape)
#     # print(surfivspred.shape)
#     err = torch.mean(100 * torch.abs((surfivspred - surfivs)/ surfivs), dim =0) 
#     avgerror = [err[x].item() for x in indices]
#     # print(torch.max(err)[0])
#     # print(err.shape)
#     # print(err)
#     plt.title("Average relative error",fontsize=15,y=1.04)
#     # plt.imshow(err.reshape(len(tlist),len(Klist)))

#     ### 04-27
#     plt.imshow(err.reshape(len(tlist),len(Klist)),origin='lower')

#     plt.colorbar(format=mtick.PercentFormatter())

#     ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
#     ax.set_xticklabels(Klist.astype('int'))

#     # ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
#     ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
#     # ax.set_yticklabels([str(round(t,2)) for t in tlist][::-1])
#     ax.set_yticklabels([str(round(t,2)) for t in tlist])
#     # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     # ax.yaxis.set_major_formatter('{x:9<5.1f}')
#     # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#     plt.xlabel("Strike",fontsize=15,labelpad=5)
#     plt.ylabel("Maturity",fontsize=15,labelpad=5)
#     rotate_xticklabels(ax, 45)

#     ax=plt.subplot(1,3,2)
#     # err = 100*np.std(np.abs((surfivspred-surfivs)/surfivs),axis = 0)
#     err = 100*torch.std(torch.abs((surfivspred-surfivs)/surfivs),dim = 0)
#     # print(torch.max(err)[0])
#     # print(err.shape)
#     stderror = [err[x].item() for x in indices]

#     plt.title("Std relative error",fontsize=15,y=1.04)
#     # origin='lower' 04-27 update
#     plt.imshow(err.reshape(len(tlist),len(Klist)),origin='lower')
#     plt.colorbar(format=mtick.PercentFormatter())
#     ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
#     ax.set_xticklabels(Klist.astype('int'))
#     ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
#     # ax.set_yticklabels([str(round(t,2)) for t in tlist][::-1])
#     ax.set_yticklabels([str(round(t,2)) for t in tlist])
#     plt.xlabel("Strike",fontsize=15,labelpad=5)
#     plt.ylabel("Maturity",fontsize=15,labelpad=5)
#     rotate_xticklabels(ax, 45)

#     ax=plt.subplot(1,3,3)
#     # err = 100*np.max(np.abs((surfivspred-surfivs)/surfivs),axis = 0)
#     err = 100*torch.max(torch.abs((surfivspred-surfivs)/surfivs),dim = 0)[0]
#     # print(torch.max(err)[0])
#     # print(err.shape)
#     maxerror = [err[x].item() for x in indices]

#     plt.title("Maximum relative error",fontsize=15,y=1.04)
#     ## origin='lower' 04-27 update
#     plt.imshow(err.reshape(len(tlist),len(Klist)),origin='lower')
#     plt.colorbar(format=mtick.PercentFormatter())
#     ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
#     ax.set_xticklabels(Klist.astype('int'))
#     ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
#     ax.set_yticklabels([str(round(t,2)) for t in tlist])
#     # ax.set_yticklabels([str(round(t,2)) for t in tlist][::-1])

#     plt.xlabel("Strike",fontsize=15,labelpad=5)
#     plt.ylabel("Maturity",fontsize=15,labelpad=5)
#     rotate_xticklabels(ax, 45)
    
#     plt.tight_layout()
#     plt.savefig(savepath, dpi=300)
    
#     # print(err)

#     # not sure
#     plt.close()
#     plt.show()
#     # print('avg', avgerror)

#     dictyml = {}
#     dictyml['indices'] = indices
#     dictyml['avgerror'] = avgerror
#     dictyml['stderror'] = stderror
#     dictyml['maxerror'] = maxerror
#     import yaml
#     with open(ymlpath, 'w') as f:
#         documents = yaml.dump(dictyml, f)

#     return indices, avgerror, stderror, maxerror

def inpainting_error(surfivs, surfivspred, Klist, tlist, savepath, mask, ymlpath):
    """surfivs shape (batch_size, image_size_y, image_size_x)"""

    def rotate_xticklabels(ax, angle):
        for tick in ax.get_xticklabels():
            tick.set_rotation(angle)

    def set_ticks_and_labels(ax, ticks, labels, rotation=45):
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=rotation)
    
    if len(Klist) <= 8:
        xticks = np.linspace(0, len(Klist) - 1, len(Klist))
        xticklabels = Klist.astype('int')
    else:
        skip = len(Klist) // 8
        xticks = np.linspace(0, len(Klist) - 1, len(Klist))[::skip]
        xticklabels = Klist.astype('int')[::skip]



    from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import torch
    # import numpy as np 

    indices = np.nonzero(~mask)
    indices = indices.tolist()
    indices = [tuple(x) for x in indices]
    # avgerror = [] 
    # stderror = []

    plt.figure(1,figsize=(14,4))
    ax=plt.subplot(1,3,1)
    # err = np.mean(100 * np.abs((surfivspred - surfivs)/ surfivs), axis =0) 
    # print(surfivs.shape)
    # print(surfivspred.shape)
    err = torch.mean(100 * torch.abs((surfivspred - surfivs)/ surfivs), dim =0) 
    avgerror = [err[x].item() for x in indices]
    # print(torch.max(err)[0])
    # print(err.shape)
    # print(err)
    plt.title("Average relative error",fontsize=15,y=1.04)
    # plt.imshow(err.reshape(len(tlist),len(Klist)))

    ### 04-27
    plt.imshow(err.reshape(len(tlist),len(Klist)),origin='lower')

    plt.colorbar(format=mtick.PercentFormatter())

    # 0504 OLD
    # ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
    # ax.set_xticklabels(Klist.astype('int'))
    # rotate_xticklabels(ax, 45)

    set_ticks_and_labels(ax, xticks, xticklabels)


    # ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    # ax.set_yticklabels([str(round(t,2)) for t in tlist][::-1])
    ax.set_yticklabels([str(round(t,2)) for t in tlist])
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter('{x:9<5.1f}')
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel("Strike",fontsize=15,labelpad=5)
    plt.ylabel("Maturity",fontsize=15,labelpad=5)
    

    ax=plt.subplot(1,3,2)
    # err = 100*np.std(np.abs((surfivspred-surfivs)/surfivs),axis = 0)
    err = 100*torch.std(torch.abs((surfivspred-surfivs)/surfivs),dim = 0)
    # print(torch.max(err)[0])
    # print(err.shape)
    stderror = [err[x].item() for x in indices]

    plt.title("Std relative error",fontsize=15,y=1.04)
    # origin='lower' 04-27 update
    plt.imshow(err.reshape(len(tlist),len(Klist)),origin='lower')
    plt.colorbar(format=mtick.PercentFormatter())

    # 0504 OLD
    # ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
    # ax.set_xticklabels(Klist.astype('int'))
    # rotate_xticklabels(ax, 45)
    set_ticks_and_labels(ax, xticks, xticklabels)


    ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    # ax.set_yticklabels([str(round(t,2)) for t in tlist][::-1])
    ax.set_yticklabels([str(round(t,2)) for t in tlist])
    plt.xlabel("Strike",fontsize=15,labelpad=5)
    plt.ylabel("Maturity",fontsize=15,labelpad=5)

    ax=plt.subplot(1,3,3)
    # err = 100*np.max(np.abs((surfivspred-surfivs)/surfivs),axis = 0)
    err = 100*torch.max(torch.abs((surfivspred-surfivs)/surfivs),dim = 0)[0]
    # print(torch.max(err)[0])
    # print(err.shape)
    maxerror = [err[x].item() for x in indices]

    plt.title("Maximum relative error",fontsize=15,y=1.04)
    ## origin='lower' 04-27 update
    plt.imshow(err.reshape(len(tlist),len(Klist)),origin='lower')
    plt.colorbar(format=mtick.PercentFormatter())

    # 0504 OLD 
    # ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
    # ax.set_xticklabels(Klist.astype('int'))
    # rotate_xticklabels(ax, 45)
    set_ticks_and_labels(ax, xticks, xticklabels)

    ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    ax.set_yticklabels([str(round(t,2)) for t in tlist])
    # ax.set_yticklabels([str(round(t,2)) for t in tlist][::-1])

    plt.xlabel("Strike",fontsize=15,labelpad=5)
    plt.ylabel("Maturity",fontsize=15,labelpad=5)
    
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    
    # print(err)

    # not sure
    plt.close()
    plt.show()
    # print('avg', avgerror)

    dictyml = {}
    dictyml['indices'] = indices
    dictyml['avgerror'] = avgerror
    dictyml['stderror'] = stderror
    dictyml['maxerror'] = maxerror
    import yaml
    with open(ymlpath, 'w') as f:
        documents = yaml.dump(dictyml, f)

    return indices, avgerror, stderror, maxerror

def tf_diff_axis_0(a):
    return a[1:]-a[:-1]

def tf_diff_axis_1(a):
    return a[:,1:]-a[:,:-1]

def tf_diff_axis_2(a):
    return a[:,:,1:]-a[:,:,:-1]

def Dcondloss_torch(y_true, y_pred, Klist, loss_multiplier=1):
    """Need to return a tensor of shape (n,) where n for the number of samples
        The shape of y_true should be (n,height, width)
    """
    # only use tf libraries for performance reason
    # x = tf.convert_to_tensor(Klist, dtype=tf.float32)
    x = torch.as_tensor(Klist, device=y_true.device)
    dx = tf_diff_axis_0(x)
    df = tf_diff_axis_2(y_pred)/dx 
    ddf = tf_diff_axis_2(df)/dx[0:-1]
    x = x[:-2]
    f = y_pred[:,:,:-2]
    df = df[:,:,:-1]
    inter1 = (1- x * df / (2 * f))**2 
    inter2 = df/4 * (1/f + 1/4)
    inter3 = ddf /2 
    Dcond = inter1 - inter2 + inter3    
    
    loss = torch.clamp(-Dcond, min = 0 ,) #  clip_value_max=tf.math.reduce_max(-Dcond))
    
    loss = torch.maximum(loss, torch.tensor(0))
    loss = torch.sum(loss, dim=1)
    loss = torch.sum(loss, dim=1)

    # maybe sum is to large? 
    Ngrid = y_true.shape[1] * y_true.shape[2]
    
#     loss = Dcondtotalvaropt(y_pred, x = Klist)
#     loss = tf.convert_to_tensor(loss, dtype=tf.float32)
    return loss_multiplier * loss / Ngrid

def calloss_torch(y_true, y_pred, tlist, loss_multiplier=1):
    """Need to return a tensor of shape (n,) where n for the number of samples
        The shape of y_true should be (n,height, width)
    """
    x = torch.as_tensor(tlist, device=y_true.device)
    dx = tf_diff_axis_0(x)
    dx = torch.reshape(dx , (1,dx.shape[0],1))
    df = tf_diff_axis_1(y_pred) / dx

    loss = torch.clamp(-df, min = 0 ,) #  clip_value_max=tf.math.reduce_max(-Dcond))

    # loss = tf.clip_by_value(-df, clip_value_min=0.0, clip_value_max=tf.math.reduce_max(-df))
    loss = torch.maximum(loss, torch.tensor(0))
    loss = torch.sum(loss, dim=1)
    loss = torch.sum(loss, dim=1)
    # maybe sum is to large? 
    Ngrid = y_true.shape[1] * y_true.shape[2]
    return loss_multiplier * loss / Ngrid



import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
# 

def plot_sampleresult(file_name, output_folder='sampleresultPlot'):
    # Load data from YAML file
    with open(file_name, 'r') as f:
        data = yaml.safe_load(f)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define variables to plot
    # variables = ['c25quantile', 'c90quantile', 'd25quantile', 'd90quantile', 'updateavg']
    variables = ['c25quantile', 'c90quantile', 'b25quantile', 'b90quantile', 'updateavg']
    ylabellist = ['25% quantile', '90% quantile','25% quantile', '90% quantile','update rate']
    noise_level_log = np.log10(data['noise_level'])

    # Plot and save figures
    for i in range(5):
        variable = variables[i]
        ylabel = ylabellist[i]
        plt.figure()
        plt.plot(noise_level_log, data[variable], marker='o')
        plt.xlabel('log10(noise level)')
        plt.ylabel(ylabel)

        # plt.title(f'{variable} vs log10(Noise Level)')
        plt.gca().invert_xaxis()  # Invert the x-axis
        plt.savefig(os.path.join(output_folder, f'{variable}_vs_noise_level.png'))
        # plt.show()
        plt.close()

# Usage:
# plot_sampleresult('sampleresult.yml')
