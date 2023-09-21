# Build Network

(shapes)=
## Rule `build_shapes`
Builds `*.geojson` files describing the interconnects in the model. Shapes are built using the Balancing Authority boundries and the offshore shapes specified in the config file. 

(base)=
## Rule `build_base_network`
Reads in [Breakthrough Energy](https://breakthroughenergy.org/) infrastructure data, and converts it into PyPSA compatiable components. A base netowork file (`*.nc`) is written out. Included in this network are: 
- Geolocated buses 
- Geoloactated AC and DC power lines 
- Transformers 

(load)=
## Rule `build_load_data`

(busregions)=
## Rule `build_bus_regions`

(renewableprofiles)=
## Rule `build_renewable_profiles`

(electricity)=
## Rule `add_electricity`
```{eval-rst}  
.. automodule:: add_electricity
```