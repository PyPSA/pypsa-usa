# Build Network

## Rule `build_shapes`
Builds `*.geojson` files describing the interconnects in the model. Shapes are built using the Balancing Authority boundries and the offshore shapes specified in the config file. 

## Rule `build_base_network`
Reads in [Breakthrough Energy](https://breakthroughenergy.org/) infrastructure data, and converts it into PyPSA compatiable components. A base netowork file (`*.nc`) is written out. Included in this network are: 
- Geolocated buses 
- Geoloactated AC and DC power lines 
- Transformers 

## Rule `build_load_data`


## Rule `build_bus_regions`


## Rule `build_renewable_profiles`


## Rule `add_electricity`