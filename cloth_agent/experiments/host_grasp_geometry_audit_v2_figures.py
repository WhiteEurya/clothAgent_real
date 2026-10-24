"""Additional static diagnostics from saved audit metrics; no fits or runtime calls."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def render_extra_figures(p):
    p=Path(p)
    allm=json.loads((p/'per_round_metrics.json').read_text())
    for n in (7,9,10):
     m=next(r for r in allm if r['global_round']==n);d=p/f'round_{n:02d}';rgb=np.asarray(Image.open(Path(m['source_directory'])/'before/rgb.png'));x,y=m['grasp_px'];s=m['nearest_candidate'];nx,ny=s['selected_pixel']
     data=np.load(d/'plane_31.npz');z=data['world_xyz'][...,2].copy();z[~data['valid']]=np.nan
     fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
     axs[0].imshow(rgb[y-20:y+21,x-20:x+21],interpolation='nearest',extent=[x-20.5,x+20.5,y+20.5,y-20.5]);axs[0].set_title('RGB, no interpolation')
     cmap=plt.colormaps['viridis'].copy();cmap.set_bad('black')
     im=axs[1].imshow(z,cmap=cmap,interpolation='nearest',extent=[x-15.5,x+15.5,y+15.5,y-15.5]);fig.colorbar(im,ax=axs[1],label='world Z (mm)');axs[1].set_title('Measured height; black=missing')
     for ax in axs[:2]:
      ax.plot(x,y,'r+',ms=18,mew=2,label='Original Pa');ax.plot(nx,ny,'o',mfc='none',mec='lime',mew=2,ms=12,label='Depth-valid candidate')
      ax.add_patch(Rectangle((x-2.5,y-2.5),5,5,fill=False,ec='red',lw=1));ax.add_patch(Rectangle((nx-2.5,ny-2.5),5,5,fill=False,ec='lime',lw=1));ax.set_xlabel('image x');ax.set_ylabel('image y')
     axs[0].legend(fontsize=8,loc='lower right')
     orig=m['grasp_depth_stats'];new=s['new_depth_stats'];axs[2].axis('off')
     text=(f'Round {n}\nOriginal {x,y}:\n  valid {orig["valid_count"]}/25\n  center valid: {orig["center_valid"]}\n  range {orig["world_z_range_mm"]:.3f} mm\n\nCandidate {nx,ny}:\n  displacement {s["selected_pixel_displacement"]:.3f} px\n  valid {new["valid_count"]}/25\n  center valid: {new["center_valid"]}\n  range {new["world_z_range_mm"]:.3f} mm\n\nFirst tested radius: 3 px\nPb unchanged; no IK/motion validation')
     axs[2].text(.02,.95,text,va='top',fontsize=11)
     fig.savefig(d/'snap_detail.png',dpi=160);plt.close(fig)
     if n in (9,10):
      data=np.load(d/'plane_05.npz');q=data['world_xyz'][data['valid']];f=m['plane_fits'][0]
      X,Y=np.meshgrid(np.linspace(q[:,0].min(),q[:,0].max(),10),np.linspace(q[:,1].min(),q[:,1].max(),10));Z=f['a']*X+f['b']*Y+f['c']
      fig=plt.figure(figsize=(8,6),layout='constrained');ax=fig.add_subplot(projection='3d')
      ax.plot_surface(X,Y,Z,alpha=.3,color='steelblue');ax.scatter(q[:,0],q[:,1],q[:,2],c='black',s=25,label='Measured XYZ')
      for pt,pred in zip(q,data['fitted_world_z'][data['valid']]):ax.plot([pt[0],pt[0]],[pt[1],pt[1]],[pt[2],pred],color='red')
      ax.set(xlabel='World X (mm)',ylabel='World Y (mm)',zlabel='World Z (mm)',title=f'Round {n}: 5x5 measured XYZ + robust plane\nred segments = vertical residuals')
      ax.legend();fig.savefig(d/'world_plane_3d.png',dpi=160);plt.close(fig)
    # IK case is visualized separately, with no plane classifier or snap repair.
    m=next(r for r in allm if r['global_round']==11)
    rgb=np.asarray(Image.open(Path(m['source_directory'])/'before/rgb.png'))
    fig,ax=plt.subplots(figsize=(12,7),layout='constrained')
    ax.imshow(rgb)
    a,b=m['grasp_px'],m['target_px']
    ax.plot(*a,'r+',ms=18,mew=2,label=f'Pa {a}')
    ax.plot(*b,'c+',ms=18,mew=2,label=f'Pb {b}')
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='yellow',lw=2))
    ax.set_title('Round 11: SEPARATE_IK_FAILURE | no depth classification / snap repair\n'
                 'IK residual 1.173 mm / 0.467 deg; required <=0.5 mm / <=0.2 deg')
    ax.axis('off');ax.legend();fig.savefig(p/'round_11/diagnostic.png',dpi=150);plt.close(fig)
