import xarray as xr
import glob
import glob
import xarray as xr
import pandas as pd
import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata, NearestNDInterpolator
import numpy as np
import time
os.chdir(r'C:\Users\user\OneDrive - NIWA\Desktop\week-it-snowed-everywhere')

# load the dataset for transfer learning
df2 = xr.open_dataset(r".\data\highresolution_northland_temperature_data.nc")
df21 = df2.reindex(time1 = df2.time1.to_index().sort_values())

sites = pd.read_csv('.\data\metadata\Site_coordsinates.csv', index_col=0)
sites = sites[['A64871', 'A42581', 'A53291', 'A64751', 'B75181', 'B75381', 'B75571',
       'B76621', 'B76951', 'B86124', 'C75731', 'C94001', 'C95251', 'D05481',
       'D05960', 'D87692', 'D87811', 'D96591', 'D96681', 'E04891', 'E05363',
       'E14272', 'E95451', 'E95681', 'E95902', 'F20791', 'F38961', 'F47691',
       'G12092', 'G12581', 'G13211', 'G13231', 'G13592', 'G13651', 'G22581',
       'H22862', 'H31352', 'H31562', 'H31971', 'H32412', 'H32552', 'H32561',
       'H32641', 'H32791', 'H40041', 'H40182', 'H41421', 'H41701', 'I50212',
       'I50852', 'I58061', 'I59161', 'I59234', 'I59361', 'I68093']]
sites = sites.T[(sites.T.lat >-40.7) & (sites.T.lon> 173.5)].T
coords = sites.values.T


# writing a function to find the nearest coordinates
def min_nearest_coords(lat, lon):
    dist = ((lats - lat)**2 + (lons - lon)**2)
    return np.where(dist == dist.min())


df22 = df21.isel(rlat = slice(25,416+25), rlon = slice(30,256 +30))
subset = df22
lats, lons = df22.latitude.values, df22.longitude.values
coords_in_arr = np.array([min_nearest_coords(*coord) for coord in coords])[:,:,0]
# finds the correct coordinates in the array

lats, lons = subset.rlat, subset.rlon
lats, lons = np.meshgrid(lats, lons)
lats, lons = lats.T, lons.T


# visualizing the sites used for this reconstruction
fig, ax = plt.subplots()
ax.contourf(df22['sfc_temp'][0])
ax.plot(coords_in_arr[:,1], coords_in_arr[:,0],'rx')
fig.show()


min_accors_time = subset.min(["rlat","rlon"])
idx = np.where(min_accors_time['sfc_temp'].values > -52)[0]
# max_accors_time = subset.max(["rlat","rlon"])
#
# fig, ax = plt.subplots()
# max_accors_time['sfc_temp'].plot(ax = ax)
# fig.show()


subset1 = subset.isel(time1 = idx)

df_stacked = xr.concat([subset1.isel(rlat = coords_in_arr.T[0,i], rlon=coords_in_arr.T[1,i]).expand_dims("z") for i in
                        range(len(coords_in_arr))],
                       dim ="z")
# checking whether this matched up
#fig, ax = plt.subplots()
#train_set['sfc_temp'].isel(x =46, y = 81).plot(ax =ax, color='b')
#subset1['sfc_temp'].isel(rlat = 46, rlon = 81, time1 = slice(59,100)).plot(ax = ax, color='r')
#fig.show()
#fig, ax = plt.subplots()



def merge_data(x, pts =[416,256]):
    print("merging")
    xi, yi = np.meshgrid(np.arange(pts[0]), np.arange(pts[1]))
    xi = xi.T
    yi = yi.T
    interp = NearestNDInterpolator(list(coords_in_arr), x.astype('int8'))
    return interp(xi, yi)#.astype('int8')
# Using nearest neigbour interpolation to initialize the outputs
"""
There is likely an issue in this part of the code
"""
""""On this line in created df_stacked is the issue
"""
train_set = xr.apply_ufunc(merge_data, df_stacked.isel(time1 = slice(0,None)),
                           input_core_dims=[["z"]],
                           output_core_dims=[["x","y"]],
                           vectorize=True, dask='parallelized')

X = train_set.astype('int8')
x = np.array(X['sfc_temp']).astype('int8')
x_train, x_test, y_train, y_test = train_test_split(np.repeat(x[:,:,:,np.newaxis],3,axis =-1),
                                                    (subset1['sfc_temp'].isel(time1 = slice(0,None)).values.astype('int8')),
                                                    test_size=0.01, shuffle=False)
# fig, ax = plt.subplots()
# # Checkig a again
# ax.plot(y_train[:150,87,124],'g')
# ax.plot(x_train[:100,87,124,0])
# fig.show()
# ip = tf.keras.layers.Input(shape = (416,256,3))
# pre = tf.keras.applications.mobilenet.preprocess_input(ip)
# bm = sm.Unet('mobilenet',input_shape =(416,256,3), encoder_weights ="imagenet",
#              activation='linear',
#              decoder_use_batchnorm = True, encoder_freeze = True)(pre)
# model1 = tf.keras.models.Model(ip, bm)
# model1.trainable = True
# model1.compile(loss ='mse', optimizer='adam')
model1 = tf.keras.models.load_model(r'C:\Users\user\OneDrive - NIWA\Desktop\week-it-snowed-everywhere\interp_unet_updated.h5')

#model1.layers[3].load_weights('resnet18_encoder_for_wind_and_decoder.h5')
#encoder_model = tf.keras.models.Model(ip, model1.layers[3].layers[85](ip))
#model1.load_weights("pretained_model_for_superers.h5")
# model.save('northland_superres_wind_model_auckland.h5')
#model.trainable = True

# pause to train the encoder
# model1.trainable = True
# for layer in model1.layers[3].layers[:110]:
#     layer.trainable = False
model1.compile(loss ='mse', optimizer='adam')
model1.fit(x_train + 128, y_train,
           validation_data=(x_test + 128, y_test),
           epochs=125, batch_size=10, shuffle =True)
model1.save(r'C:\Users\user\OneDrive - NIWA\Desktop\week-it-snowed-everywhere\interp_unet_updated1.h5')
model1.save_weights('model_updated.1h5')
#model1.save(r'historical_model_daily_min.h5')

preds = model1.predict(x_test + 128, verbose =1,batch_size =1)

# fig, ax = plt.subplots(2)
# # ax.plot(preds[:150,50,150,0])
# # ax.plot(x_test[:150,50,150],'r-')
# train_set['sfc_temp'].isel(time1 = 0).plot(ax = ax[0])
# subset1['sfc_temp'].isel(time1 = 0).plot(ax = ax[1])
# fig.show()
#
# preds = model1.predict(x_test + 128, verbose =1,batch_size =1)
#
# fig, ax = plt.subplots(2, figsize =(10,15))
# ax[0].contourf(preds[88,:,:,0], cmap ='RdBu_r',
#             levels =np.arange(-50, 128, 1), extend ='both')
# ax[1].contourf(y_test[88,:,:], cmap ='RdBu_r',
#              levels = np.arange(-50, 128, 1), extend ='both')


fig.show()
fig.savefig('Extremes_on_a_given_day.png')

model1.save_weights(r'C:\Users\user\OneDrive - NIWA\Desktop\week-it-snowed-everywhere\interp_unet.h5')
import cartopy.crs as ccrs
proj = ccrs.RotatedPole(pole_latitude=49.55, pole_longitude=171.77, central_rotated_longitude=180)
#
transform = ccrs.RotatedPole(pole_latitude=49.55, pole_longitude=171.77, central_rotated_longitude=0)
fig, (ax ,ax2)= plt.subplots(1,2, subplot_kw=dict(projection = proj))
ax.contourf(lons, lats,preds[15,:,:,0], cmap='jet', transform = transform, levels = np.arange(-10, 80, 1), extend ='both')
#fig.show()


#fig, ax = plt.subplots()
ax2.contourf(lons, lats,y_test[15,:,:], cmap='jet', transform = transform, levels = np.arange(-10, 80, 1), extend ='both')
ax2.coastlines('10m')
ax.coastlines('10m')
# z = np.zeros([224,224]) * np.nan
# z = z.ravel()
# z[random_pts.astype('int32')] =1
# z = z.reshape(224,224)
# x1, y1 = np.where(z>0.0)
ax2.plot(lons[coords_in_arr[:,0],coords_in_arr[:,1]],lats[coords_in_arr[:,0],coords_in_arr[:,1]],'ko', markersize =12, transform = transform)
fig.show()


prediction_arr = xr.open_dataset(r"C:\Users\user\OneDrive - NIWA\Desktop\week-it-snowed-everywhere\data\the_week_it_snowed_gridded_deg.nc")
#prediction_arr = (prediction_arr -32)/1.8 + 273.15

# notes ad 128 for training



preds_data = ((255 * (prediction_arr - 230)/(320 - 250) - 128))
#
prediction = model1.predict(np.repeat(preds_data['data'].values[:,:,:,np.newaxis].astype('int8'),3, axis =-1) + 128)
#prediction = model1.predict(preds_data.expand_dims({"channel":3})['data'].values.transpose(1,2,3,0) +128)
preds_ = ((prediction + 128) * (320 - 250)/255.0 + 230 - 273.15)
preds2 = ((preds_data['data'].values + 128) * (320 - 250)/255.0 + 230 - 273.15) #*1.15 +0.34

import cartopy.crs as ccrs
preds_data['data'].values = preds_[:,:,:,0]
average_temp = preds_data.groupby(preds_data.time.dt.month).min()
fig, ax = plt.subplots(1,3, subplot_kw=dict(projection = proj), figsize = (10,15))
ax = ax.ravel()
for i in range(3):
    cs = ax[i].contourf(lons, lats, average_temp.isel(month = i)['data'].values, cmap='jet', transform=transform, levels=np.arange(-5, 5, 0.05),
                     extend='both')
    #ax[i].set_title(average_temp.time.to_index().strftime("%Y-%B")[i])
fig.show()




proj = ccrs.RotatedPole(pole_latitude=49.55, pole_longitude=171.77, central_rotated_longitude=180)
#
transform = ccrs.RotatedPole(pole_latitude=49.55, pole_longitude=171.77, central_rotated_longitude=0)
fig, ax= plt.subplots(1,1, subplot_kw=dict(projection = proj))
cs = ax.contourf(lons, lats,preds_[59,:,:,0], cmap='jet', transform = transform, levels = np.arange(-5, 6, 0.1), extend ='both')
#fig.show()
ax.coastlines('10m')
fig.colorbar(cs, ax = ax)
fig.show()


fig, ax = plt.subplots()
ax.plot(preds_[:, 397,28,0], color ='r')
ax.plot(preds2[:, 397,28], color ='b')
#ax.plot(preds_[:100, 397, 28,0], color ='r')
#ax.plot(prediction_arr['data'][:100, 397, ] - 273.15)
fig.show()

meanx = np.nanmean(x_train[:], axis =(0,-1))
meanp = preds_data.mean("time")['data'].values
fig, ax = plt.subplots(

)
ax.plot(meanx.ravel(), meanp.ravel(),'rx')
fig.show()

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# index =3
# proj = ccrs.RotatedPole(pole_latitude=49.55, pole_longitude=171.77, central_rotated_longitude=180)
# #
# transform = ccrs.RotatedPole(pole_latitude=49.55, pole_longitude=171.77, central_rotated_longitude=0)
#
# times = df.time0[-len(x_test):].to_index()
# fig, ax = plt.subplots(2,1,
#                        subplot_kw=dict(projection = proj), figsize = (8,12)
#                        ,sharex = True, sharey =True)
# cs2 = ax[0].contourf(lons,
#                lats,preds[index,:,:,0], cmap ='jet', transform =transform, extend ='both', levels = np.arange(0,0.25,0.01))
# cs = ax[1].contourf(lons,
#                lats,y_test[index], cmap ='jet',
#                transform =transform, extend ='both', levels = np.arange(0,0.25,0.01))
# ax[0].coastlines('10m')
# ax[0].set_title('Downscaled')
# ax[1].set_title('NZCSM Forecasted')
# ax[1].coastlines('10m')
# fig.suptitle(f'Downscaled Maximum Daily Wind Gust Using Machine Learning \n'
#              f'at {times[index].strftime("%Y-%m-%d")}')
#
# cbar = fig.colorbar(cs, ax=ax[1])
# cbar2 = fig.colorbar(cs2, ax=ax[0])
# cbar.set_label('Daily Maximum Wind Gust (m/s)')
# cbar2.set_label('Daily Maximum Wind Gust (m/s)')
#
#
# fig.show()
# lats = df2.latitude
# lons = df2.longitude
# lats, lons = np.meshgrid(lats, lons)
# lats = lats.T
# lons = lons.T
#
# z =df2['u'].sel(time = pd.to_datetime(times[index].strftime("%Y-%m-%d"))).values
# fig, ax = plt.subplots(3,1,sharex = True, sharey =True,
#                        subplot_kw=dict(projection = ccrs.PlateCarree(central_longitude=171.77)), figsize = (8,12))
# z2 =df2['v'].sel(time = pd.to_datetime(times[index].strftime("%Y-%m-%d"))).values
# z3 =df2['w'].sel(time = pd.to_datetime(times[index].strftime("%Y-%m-%d"))).values
# ax = ax.ravel()
# cs1 = ax[0].contourf(lons, lats,z,
#                  transform = ccrs.PlateCarree(), cmap='jet', levels =np.arange(5,18,0.5))
# cs2 = ax[1].contourf(lons, lats,z2,
#                  transform = ccrs.PlateCarree(), cmap='jet', levels =np.arange(-18,0,0.5))
# cs3 = ax[2].contourf(lons, lats,z3,
#                  transform = ccrs.PlateCarree(),  levels =np.arange(-1,1,0.1), cmap='jet')
# cbar1 = fig.colorbar(cs1, ax =ax[0])
# cbar2 = fig.colorbar(cs2, ax =ax[1])
# cbar3 = fig.colorbar(cs3, ax =ax[2])
# ax[2].set_extent([173,177,-35,-38])
# cbar3.set_label('W$_{850}$hPa (m/s)')
# cbar2.set_label('V$_{850}$hPa (m/s)')
# cbar1.set_label('U$_{850}$hPa (m/s)')
# ax[2].coastlines('10m')
# ax[0].coastlines('10m')
# ax[1].coastlines('10m')
# ax[0].set_title('U$_{850}$hPa')
# ax[1].set_title('V$_{850}$hPa')
# ax[2].set_title('W$_{850}$hPa')
# fig.show()
#
#
# fig.savefig('Predicto_variables_for_frame16.png', dpi =300)
#
#
#
# df1 = pd.DataFrame(index = times)
# df1['Downscaled'] = preds[:,500,100]
# df1['NZCSM'] =y_test[:,500,100]
# fig,ax =plt.subplots()
# df1.plot(ax =ax )
# ax.set_title('Downscaled Maximum Daily Wind Gust Using Machine Learning \n'
#              'At a Site in Auckland')
# ax.set_ylabel('Wind Speed (m/s)')
# fig.show()
# fig.savefig('NZCSM_downscaled_site.png', dpi =300)
# x1 =[]
# for i in range(y_test.shape[0]):
#     z = y_test[i].ravel()
#     y = preds[i].ravel()
#     x1.append(np.corrcoef(z, y)[0,1])
#
# import numpy as np
# fig, ax = plt.subplots(subplot_kw=dict(projection = ccrs.PlateCarree(central_longitude=171.77)))
# monthly = df['uwnd'].resample(time = '1MS').mean()
# def demean(x):
#     return x -x.mean("time")
# anoms = monthly#.groupby(monthly.time.dt.month).apply(demean)
# anoms.isel(time = 6).sel(level =200).plot.contourf(ax = ax, transform = ccrs.PlateCarree(),
#                                                        levels = np.arange(-50,55,5),
#                                                        extend ='both')
# ax.coastlines('110m')
# fig.show()
