import xarray as xr

ds = xr.open_dataset('/home/user/clm-ml-jax/tests/mixing_length_test/sample_data/bomex_mixing_length_calculation_samples.nc')
sample = ds.isel(samples=0)

sample = sample.expand_dims(dim="column", axis=0)

print("Fortran Lscale_down pattern:")
print("k   zt[m]  Lscale_down[m]  alt_reached[m]")
print("-" * 50)
for k in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
    zt_val = float(sample.zt[k])
    lscale_down = float(sample.Lscale_down[0, k])
    alt_reached = zt_val - lscale_down
    print(f"{k:2d}  {zt_val:6.1f}  {lscale_down:14.2f}  {alt_reached:14.1f}")
