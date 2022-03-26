import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
sns.set_context(context='talk',font_scale=0.7)

def plot_demo_obs(ds_obs, central_date, delta_t):

    tmin = central_date - delta_t
    tmax = central_date + delta_t
    
    ds_obs = ds_obs.sel(time=slice(tmin, tmax))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    pts = ax.scatter(ds_obs.longitude , ds_obs.latitude, c=ds_obs.sla_unfiltered, s=20, cmap='gist_stern')
    ax.add_patch(Rectangle((295, 33), 10, 10, fill=None, alpha=1))
    plt.xlabel('Longitude', fontweight='bold')
    plt.ylabel('Latitude', fontweight='bold')
    plt.title(f'Sea Level Anomaly (altimeter track)')
    plt.colorbar(pts, orientation='horizontal')
    plt.text(298, 43.5,'STUDY AREA', fontsize=20)
    plt.show()
    return fig, ax