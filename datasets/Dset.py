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

    ax.set_xlabel('log moneyness')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied volatility')

    ax.set_title(plotname + ' IVS')
    plt.savefig(savepath)
    plt.close()
    # plt.show()

    # return Z 

def inpainting_error(surfivs, surfivspred, Klist, tlist, savepath, ):
    """surfivs shape (batch_size, image_size_y, image_size_x)"""
    from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import torch
    plt.figure(1,figsize=(14,4))
    ax=plt.subplot(1,3,1)
    # err = np.mean(100 * np.abs((surfivspred - surfivs)/ surfivs), axis =0) 
    # print(surfivs.shape)
    # print(surfivspred.shape)
    err = torch.mean(100 * torch.abs((surfivspred - surfivs)/ surfivs), dim =0) 
    plt.title("Average relative error",fontsize=15,y=1.04)
    plt.imshow(err.reshape(len(tlist),len(Klist)))
    plt.colorbar(format=mtick.PercentFormatter())

    ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
    ax.set_xticklabels(Klist.astype('int'))

    # ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    ax.set_yticklabels([str(round(t,2)) for t in tlist])
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter('{x:9<5.1f}')
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xlabel("Strike",fontsize=15,labelpad=5)
    plt.ylabel("Maturity",fontsize=15,labelpad=5)

    ax=plt.subplot(1,3,2)
    # err = 100*np.std(np.abs((surfivspred-surfivs)/surfivs),axis = 0)
    err = 100*torch.std(torch.abs((surfivspred-surfivs)/surfivs),dim = 0)
    plt.title("Std relative error",fontsize=15,y=1.04)
    plt.imshow(err.reshape(len(tlist),len(Klist)))
    plt.colorbar(format=mtick.PercentFormatter())
    ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
    ax.set_xticklabels(Klist.astype('int'))
    ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    ax.set_yticklabels([str(round(t,2)) for t in tlist])
    plt.xlabel("Strike",fontsize=15,labelpad=5)
    plt.ylabel("Maturity",fontsize=15,labelpad=5)

    ax=plt.subplot(1,3,3)
    # err = 100*np.max(np.abs((surfivspred-surfivs)/surfivs),axis = 0)
    err = 100*torch.max(torch.abs((surfivspred-surfivs)/surfivs),dim = 0)[0]
    plt.title("Maximum relative error",fontsize=15,y=1.04)
    plt.imshow(err.reshape(len(tlist),len(Klist)))
    plt.colorbar(format=mtick.PercentFormatter())
    ax.set_xticks(np.linspace(0,len(Klist)-1,len(Klist)))
    ax.set_xticklabels(Klist.astype('int'))
    ax.set_yticks(np.linspace(0,len(tlist)-1,len(tlist)))
    ax.set_yticklabels([str(round(t,2)) for t in tlist])
    plt.xlabel("Strike",fontsize=15,labelpad=5)
    plt.ylabel("Maturity",fontsize=15,labelpad=5)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    # plt.show()