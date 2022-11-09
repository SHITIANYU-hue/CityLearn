Search.setIndex({docnames:["api/citylearn","api/citylearn.agents","api/citylearn.agents.base","api/citylearn.agents.rbc","api/citylearn.agents.rlc","api/citylearn.agents.sac","api/citylearn.base","api/citylearn.building","api/citylearn.citylearn","api/citylearn.citylearn_pettingzoo","api/citylearn.cost_function","api/citylearn.data","api/citylearn.energy_model","api/citylearn.preprocessing","api/citylearn.rendering","api/citylearn.reward_function","api/citylearn.rl","api/citylearn.simulator","api/citylearn.utilities","api/modules","citylearn_challenge/2020","citylearn_challenge/2021","citylearn_challenge/2022","citylearn_challenge/years","environment","index","usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/citylearn.rst","api/citylearn.agents.rst","api/citylearn.agents.base.rst","api/citylearn.agents.rbc.rst","api/citylearn.agents.rlc.rst","api/citylearn.agents.sac.rst","api/citylearn.base.rst","api/citylearn.building.rst","api/citylearn.citylearn.rst","api/citylearn.citylearn_pettingzoo.rst","api/citylearn.cost_function.rst","api/citylearn.data.rst","api/citylearn.energy_model.rst","api/citylearn.preprocessing.rst","api/citylearn.rendering.rst","api/citylearn.reward_function.rst","api/citylearn.rl.rst","api/citylearn.simulator.rst","api/citylearn.utilities.rst","api/modules.rst","citylearn_challenge/2020.rst","citylearn_challenge/2021.rst","citylearn_challenge/2022.rst","citylearn_challenge/years.rst","environment.rst","index.rst","usage.rst"],objects:{"":[[0,0,0,"-","citylearn"]],"citylearn.agents":[[2,0,0,"-","base"],[3,0,0,"-","rbc"],[4,0,0,"-","rlc"],[5,0,0,"-","sac"]],"citylearn.agents.base":[[2,1,1,"","Agent"]],"citylearn.agents.base.Agent":[[2,2,1,"","action_dimension"],[2,2,1,"","action_space"],[2,2,1,"","actions"],[2,3,1,"","add_to_buffer"],[2,2,1,"","building_information"],[2,3,1,"","next_time_step"],[2,2,1,"","observation_names"],[2,2,1,"","observation_space"],[2,3,1,"","reset"],[2,3,1,"","select_actions"],[2,3,1,"","set_encoders"]],"citylearn.agents.rbc":[[3,1,1,"","BasicBatteryRBC"],[3,1,1,"","BasicRBC"],[3,1,1,"","OptimizedRBC"],[3,1,1,"","RBC"]],"citylearn.agents.rbc.BasicBatteryRBC":[[3,3,1,"","select_actions"]],"citylearn.agents.rbc.BasicRBC":[[3,3,1,"","select_actions"]],"citylearn.agents.rbc.OptimizedRBC":[[3,3,1,"","select_actions"]],"citylearn.agents.rlc":[[4,1,1,"","RLC"]],"citylearn.agents.rlc.RLC":[[4,2,1,"","action_scaling_coefficient"],[4,2,1,"","alpha"],[4,2,1,"","batch_size"],[4,2,1,"","deterministic_start_time_step"],[4,2,1,"","discount"],[4,2,1,"","end_exploration_time_step"],[4,2,1,"","hidden_dimension"],[4,2,1,"","lr"],[4,2,1,"","observation_dimension"],[4,2,1,"","replay_buffer_capacity"],[4,2,1,"","reward_scaling"],[4,2,1,"","seed"],[4,2,1,"","start_training_time_step"],[4,2,1,"","tau"],[4,2,1,"","update_per_time_step"]],"citylearn.agents.sac":[[5,1,1,"","SAC"],[5,1,1,"","SACBasicBatteryRBC"],[5,1,1,"","SACBasicRBC"],[5,1,1,"","SACOptimizedRBC"],[5,1,1,"","SACRBC"]],"citylearn.agents.sac.SAC":[[5,3,1,"","add_to_buffer"],[5,3,1,"","get_encoded_observations"],[5,3,1,"","get_exploration_actions"],[5,3,1,"","get_normalized_observations"],[5,3,1,"","get_normalized_reward"],[5,3,1,"","get_post_exploration_actions"],[5,3,1,"","select_actions"],[5,3,1,"","set_encoders"],[5,3,1,"","set_networks"]],"citylearn.agents.sac.SACRBC":[[5,3,1,"","get_exploration_actions"],[5,2,1,"","rbc"]],"citylearn.base":[[6,1,1,"","Environment"]],"citylearn.base.Environment":[[6,3,1,"","next_time_step"],[6,3,1,"","reset"],[6,3,1,"","reset_time_step"],[6,2,1,"","seconds_per_time_step"],[6,2,1,"","time_step"],[6,2,1,"","uid"]],"citylearn.building":[[7,1,1,"","Building"]],"citylearn.building.Building":[[7,2,1,"","action_metadata"],[7,2,1,"","action_space"],[7,2,1,"","active_actions"],[7,2,1,"","active_observations"],[7,3,1,"","apply_actions"],[7,3,1,"","autosize_cooling_device"],[7,3,1,"","autosize_cooling_storage"],[7,3,1,"","autosize_dhw_device"],[7,3,1,"","autosize_dhw_storage"],[7,3,1,"","autosize_electrical_storage"],[7,3,1,"","autosize_heating_device"],[7,3,1,"","autosize_heating_storage"],[7,3,1,"","autosize_pv"],[7,2,1,"","carbon_intensity"],[7,2,1,"","cooling_demand"],[7,2,1,"","cooling_device"],[7,2,1,"","cooling_electricity_consumption"],[7,2,1,"","cooling_storage"],[7,2,1,"","cooling_storage_electricity_consumption"],[7,2,1,"","dhw_demand"],[7,2,1,"","dhw_device"],[7,2,1,"","dhw_electricity_consumption"],[7,2,1,"","dhw_storage"],[7,2,1,"","dhw_storage_electricity_consumption"],[7,2,1,"","electrical_storage"],[7,2,1,"","electrical_storage_electricity_consumption"],[7,2,1,"","energy_from_cooling_device"],[7,2,1,"","energy_from_cooling_device_to_cooling_storage"],[7,2,1,"","energy_from_cooling_storage"],[7,2,1,"","energy_from_dhw_device"],[7,2,1,"","energy_from_dhw_device_to_dhw_storage"],[7,2,1,"","energy_from_dhw_storage"],[7,2,1,"","energy_from_electrical_storage"],[7,2,1,"","energy_from_heating_device"],[7,2,1,"","energy_from_heating_device_to_heating_storage"],[7,2,1,"","energy_from_heating_storage"],[7,2,1,"","energy_simulation"],[7,2,1,"","energy_to_electrical_storage"],[7,3,1,"","estimate_action_space"],[7,3,1,"","estimate_observation_space"],[7,2,1,"","heating_demand"],[7,2,1,"","heating_device"],[7,2,1,"","heating_electricity_consumption"],[7,2,1,"","heating_storage"],[7,2,1,"","heating_storage_electricity_consumption"],[7,2,1,"","name"],[7,2,1,"","net_electricity_consumption"],[7,2,1,"","net_electricity_consumption_emission"],[7,2,1,"","net_electricity_consumption_price"],[7,2,1,"","net_electricity_consumption_without_storage"],[7,2,1,"","net_electricity_consumption_without_storage_and_pv"],[7,2,1,"","net_electricity_consumption_without_storage_and_pv_emission"],[7,2,1,"","net_electricity_consumption_without_storage_and_pv_price"],[7,2,1,"","net_electricity_consumption_without_storage_emission"],[7,2,1,"","net_electricity_consumption_without_storage_price"],[7,3,1,"","next_time_step"],[7,2,1,"","non_shiftable_load_demand"],[7,2,1,"","observation_metadata"],[7,2,1,"","observation_space"],[7,2,1,"","observations"],[7,2,1,"","pricing"],[7,2,1,"","pv"],[7,3,1,"","reset"],[7,2,1,"","solar_generation"],[7,3,1,"","update_cooling"],[7,3,1,"","update_dhw"],[7,3,1,"","update_electrical_storage"],[7,3,1,"","update_heating"],[7,3,1,"","update_variables"],[7,2,1,"","weather"]],"citylearn.citylearn":[[8,1,1,"","CityLearnEnv"],[8,4,1,"","Error"],[8,4,1,"","UnknownSchemaError"]],"citylearn.citylearn.CityLearnEnv":[[8,2,1,"","action_space"],[8,2,1,"","buildings"],[8,2,1,"","central_agent"],[8,2,1,"","cooling_demand"],[8,2,1,"","cooling_electricity_consumption"],[8,2,1,"","cooling_storage_electricity_consumption"],[8,2,1,"","dhw_demand"],[8,2,1,"","dhw_electricity_consumption"],[8,2,1,"","dhw_storage_electricity_consumption"],[8,2,1,"","done"],[8,2,1,"","electrical_storage_electricity_consumption"],[8,2,1,"","energy_from_cooling_device"],[8,2,1,"","energy_from_cooling_device_to_cooling_storage"],[8,2,1,"","energy_from_cooling_storage"],[8,2,1,"","energy_from_dhw_device"],[8,2,1,"","energy_from_dhw_device_to_dhw_storage"],[8,2,1,"","energy_from_dhw_storage"],[8,2,1,"","energy_from_electrical_storage"],[8,2,1,"","energy_from_heating_device"],[8,2,1,"","energy_from_heating_device_to_heating_storage"],[8,2,1,"","energy_from_heating_storage"],[8,2,1,"","energy_to_electrical_storage"],[8,3,1,"","evaluate"],[8,3,1,"","get_building_information"],[8,3,1,"","get_default_shared_observations"],[8,3,1,"","get_info"],[8,3,1,"","get_reward"],[8,2,1,"","heating_demand"],[8,2,1,"","heating_electricity_consumption"],[8,2,1,"","heating_storage_electricity_consumption"],[8,3,1,"","load_agent"],[8,2,1,"","net_electricity_consumption"],[8,2,1,"","net_electricity_consumption_emission"],[8,2,1,"","net_electricity_consumption_price"],[8,2,1,"","net_electricity_consumption_without_storage"],[8,2,1,"","net_electricity_consumption_without_storage_and_pv"],[8,2,1,"","net_electricity_consumption_without_storage_and_pv_emission"],[8,2,1,"","net_electricity_consumption_without_storage_and_pv_price"],[8,2,1,"","net_electricity_consumption_without_storage_emission"],[8,2,1,"","net_electricity_consumption_without_storage_price"],[8,3,1,"","next_time_step"],[8,2,1,"","non_shiftable_load_demand"],[8,2,1,"","observation_names"],[8,2,1,"","observation_space"],[8,2,1,"","observations"],[8,3,1,"","render"],[8,3,1,"","reset"],[8,2,1,"","reward_function"],[8,2,1,"","rewards"],[8,2,1,"","schema"],[8,2,1,"","shared_observations"],[8,2,1,"","solar_generation"],[8,3,1,"","step"],[8,2,1,"","time_steps"],[8,3,1,"","update_variables"]],"citylearn.citylearn_pettingzoo":[[9,1,1,"","CityLearnPettingZooEnv"],[9,6,1,"","main"],[9,6,1,"","make_citylearn_env"],[9,6,1,"","raw_env"]],"citylearn.citylearn_pettingzoo.CityLearnPettingZooEnv":[[9,3,1,"","action_space"],[9,5,1,"","agents"],[9,3,1,"","close"],[9,5,1,"","metadata"],[9,3,1,"","observation_space"],[9,5,1,"","possible_agents"],[9,3,1,"","render"],[9,3,1,"","reset"],[9,3,1,"","step"]],"citylearn.cost_function":[[10,1,1,"","CostFunction"]],"citylearn.cost_function.CostFunction":[[10,3,1,"","average_daily_peak"],[10,3,1,"","carbon_emissions"],[10,3,1,"","load_factor"],[10,3,1,"","net_electricity_consumption"],[10,3,1,"","peak_demand"],[10,3,1,"","price"],[10,3,1,"","quadratic"],[10,3,1,"","ramping"]],"citylearn.data":[[11,1,1,"","CarbonIntensity"],[11,1,1,"","DataSet"],[11,1,1,"","EnergySimulation"],[11,1,1,"","Pricing"],[11,1,1,"","Weather"]],"citylearn.data.CarbonIntensity":[[11,5,1,"","carbon_intensity"]],"citylearn.data.DataSet":[[11,3,1,"","copy"],[11,3,1,"","get_names"],[11,3,1,"","get_schema"]],"citylearn.data.EnergySimulation":[[11,5,1,"","average_unment_cooling_setpoint_difference"],[11,5,1,"","cooling_demand"],[11,5,1,"","day_type"],[11,5,1,"","daylight_savings_status"],[11,5,1,"","dhw_demand"],[11,5,1,"","heating_demand"],[11,5,1,"","hour"],[11,5,1,"","indoor_dry_bulb_temperature"],[11,5,1,"","indoor_relative_humidity"],[11,5,1,"","month"],[11,5,1,"","non_shiftable_load"],[11,5,1,"","solar_generation"]],"citylearn.data.Pricing":[[11,5,1,"","electricity_pricing"],[11,5,1,"","electricity_pricing_predicted_12h"],[11,5,1,"","electricity_pricing_predicted_24h"],[11,5,1,"","electricity_pricing_predicted_6h"]],"citylearn.data.Weather":[[11,5,1,"","diffuse_solar_irradiance"],[11,5,1,"","diffuse_solar_irradiance_predicted_12h"],[11,5,1,"","diffuse_solar_irradiance_predicted_24h"],[11,5,1,"","diffuse_solar_irradiance_predicted_6h"],[11,5,1,"","direct_solar_irradiance"],[11,5,1,"","direct_solar_irradiance_predicted_12h"],[11,5,1,"","direct_solar_irradiance_predicted_24h"],[11,5,1,"","direct_solar_irradiance_predicted_6h"],[11,5,1,"","outdoor_dry_bulb_temperature"],[11,5,1,"","outdoor_dry_bulb_temperature_predicted_12h"],[11,5,1,"","outdoor_dry_bulb_temperature_predicted_24h"],[11,5,1,"","outdoor_dry_bulb_temperature_predicted_6h"],[11,5,1,"","outdoor_relative_humidity"],[11,5,1,"","outdoor_relative_humidity_predicted_12h"],[11,5,1,"","outdoor_relative_humidity_predicted_24h"],[11,5,1,"","outdoor_relative_humidity_predicted_6h"]],"citylearn.energy_model":[[12,1,1,"","Battery"],[12,1,1,"","Device"],[12,1,1,"","ElectricDevice"],[12,1,1,"","ElectricHeater"],[12,1,1,"","HeatPump"],[12,1,1,"","PV"],[12,1,1,"","StorageDevice"],[12,1,1,"","StorageTank"]],"citylearn.energy_model.Battery":[[12,2,1,"","capacity"],[12,2,1,"","capacity_history"],[12,2,1,"","capacity_loss_coefficient"],[12,2,1,"","capacity_power_curve"],[12,3,1,"","charge"],[12,3,1,"","degrade"],[12,2,1,"","efficiency"],[12,2,1,"","efficiency_history"],[12,2,1,"","electricity_consumption"],[12,3,1,"","get_current_efficiency"],[12,3,1,"","get_max_input_power"],[12,3,1,"","get_max_output_power"],[12,2,1,"","power_efficiency_curve"],[12,3,1,"","reset"]],"citylearn.energy_model.Device":[[12,2,1,"","efficiency"]],"citylearn.energy_model.ElectricDevice":[[12,2,1,"","available_nominal_power"],[12,2,1,"","electricity_consumption"],[12,3,1,"","next_time_step"],[12,2,1,"","nominal_power"],[12,3,1,"","reset"],[12,3,1,"","update_electricity_consumption"]],"citylearn.energy_model.ElectricHeater":[[12,3,1,"","autosize"],[12,2,1,"","efficiency"],[12,3,1,"","get_input_power"],[12,3,1,"","get_max_output_power"]],"citylearn.energy_model.HeatPump":[[12,3,1,"","autosize"],[12,2,1,"","efficiency"],[12,3,1,"","get_cop"],[12,3,1,"","get_input_power"],[12,3,1,"","get_max_output_power"],[12,2,1,"","target_cooling_temperature"],[12,2,1,"","target_heating_temperature"]],"citylearn.energy_model.PV":[[12,3,1,"","autosize"],[12,3,1,"","get_generation"]],"citylearn.energy_model.StorageDevice":[[12,3,1,"","autosize"],[12,2,1,"","capacity"],[12,3,1,"","charge"],[12,2,1,"","efficiency"],[12,2,1,"","efficiency_scaling"],[12,2,1,"","energy_balance"],[12,2,1,"","initial_soc"],[12,2,1,"","loss_coefficient"],[12,3,1,"","reset"],[12,3,1,"","set_energy_balance"],[12,2,1,"","soc"],[12,2,1,"","soc_init"]],"citylearn.energy_model.StorageTank":[[12,3,1,"","charge"],[12,2,1,"","max_input_power"],[12,2,1,"","max_output_power"]],"citylearn.preprocessing":[[13,1,1,"","Encoder"],[13,1,1,"","NoNormalization"],[13,1,1,"","Normalize"],[13,1,1,"","OnehotEncoding"],[13,1,1,"","PeriodicNormalization"],[13,1,1,"","RemoveFeature"]],"citylearn.rendering":[[14,1,1,"","RenderBuilding"],[14,6,1,"","get_background"],[14,6,1,"","get_plots"]],"citylearn.rendering.RenderBuilding":[[14,3,1,"","draw_building"],[14,3,1,"","draw_line"],[14,3,1,"","get_coords"],[14,3,1,"","read_glow_image"],[14,3,1,"","read_image"]],"citylearn.reward_function":[[15,1,1,"","IndependentSACReward"],[15,1,1,"","MARL"],[15,1,1,"","RewardFunction"]],"citylearn.reward_function.IndependentSACReward":[[15,3,1,"","calculate"]],"citylearn.reward_function.MARL":[[15,3,1,"","calculate"]],"citylearn.reward_function.RewardFunction":[[15,2,1,"","agent_count"],[15,3,1,"","calculate"],[15,2,1,"","carbon_emission"],[15,2,1,"","electricity_consumption"],[15,2,1,"","electricity_price"],[15,2,1,"","kwargs"]],"citylearn.rl":[[16,1,1,"","PolicyNetwork"],[16,1,1,"","RegressionBuffer"],[16,1,1,"","ReplayBuffer"],[16,1,1,"","SoftQNetwork"]],"citylearn.rl.PolicyNetwork":[[16,3,1,"","forward"],[16,3,1,"","sample"],[16,3,1,"","to"],[16,5,1,"","training"]],"citylearn.rl.RegressionBuffer":[[16,3,1,"","push"]],"citylearn.rl.ReplayBuffer":[[16,3,1,"","push"],[16,3,1,"","sample"]],"citylearn.rl.SoftQNetwork":[[16,3,1,"","forward"],[16,5,1,"","training"]],"citylearn.simulator":[[17,1,1,"","Simulator"]],"citylearn.simulator.Simulator":[[17,2,1,"","agent"],[17,2,1,"","citylearn_env"],[17,2,1,"","episodes"],[17,3,1,"","simulate"]],"citylearn.utilities":[[18,6,1,"","read_json"],[18,6,1,"","write_json"]],citylearn:[[1,0,0,"-","agents"],[6,0,0,"-","base"],[7,0,0,"-","building"],[8,0,0,"-","citylearn"],[9,0,0,"-","citylearn_pettingzoo"],[10,0,0,"-","cost_function"],[11,0,0,"-","data"],[12,0,0,"-","energy_model"],[13,0,0,"-","preprocessing"],[14,0,0,"-","rendering"],[15,0,0,"-","reward_function"],[16,0,0,"-","rl"],[17,0,0,"-","simulator"],[18,0,0,"-","utilities"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","property","Python property"],"3":["py","method","Python method"],"4":["py","exception","Python exception"],"5":["py","attribute","Python attribute"],"6":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:property","3":"py:method","4":"py:exception","5":"py:attribute","6":"py:function"},terms:{"0":[3,6,7,10,11,12,13,15,16,20,21,24,25],"00":3,"002":25,"003":16,"02":3,"03":3,"04":3,"06":[3,16],"07":3,"08":3,"1":[3,6,7,8,10,11,12,13,15,16,20,21,24,25],"10":[3,15,25],"100":10,"1000":12,"10000":10,"1016":25,"10504":25,"1072":25,"1089":25,"11":[3,25],"1100":10,"1145":[15,25],"1150":16,"12":[11,24],"15":12,"1500":10,"15th":20,"170":25,"179":25,"1800":10,"19":25,"1913":16,"1914":16,"1e":16,"2":[3,10,11,12,13,16,21,24,25],"20":[7,12,16],"200":10,"2012":25,"2018":25,"2019":25,"2020":[15,23,25],"2021":23,"2022":[8,23,24],"2324":16,"2325":16,"235":25,"2382":16,"24":[10,11,24],"273":12,"3":[3,13,15,16,20,25],"300":[10,16],"3360322":25,"3360998":25,"3408308":[15,25],"3420":16,"3427604":[15,25],"356":25,"357":25,"3741":16,"4":[3,13,25],"400":[10,16],"4443":16,"450000":10,"48550":25,"4d":16,"5":[3,7,20,21],"500":10,"50000":10,"5112":16,"5113":16,"532":3,"5593":16,"6":[3,11,24],"600":10,"610000":10,"6122":16,"6th":25,"7":[3,11,24],"700":10,"700000":10,"730":10,"7th":25,"8":[3,11],"8760":10,"9":3,"90000":10,"boolean":[8,13,24],"case":8,"class":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,24],"default":[7,8,9,10,12,15,21],"do":21,"final":20,"float":[2,3,4,5,6,7,8,10,11,12,13,15,16,24],"function":[7,8,9,10,12,16,20,21],"import":[20,24],"int":[2,4,5,6,8,10,11,13,15,17,24],"j\u00e9r\u00f4me":25,"jos\u00e9":[15,25],"k\u00e4mpf":25,"long":20,"new":[12,20,25],"return":[2,3,5,7,8,9,10,12,15,16,18,21,24],"static":[8,10,11],"super":24,"true":[7,8,9,12,16],"v\u00e1zquez":[15,25],"while":[7,12,16,24],"zolt\u00e1n":15,A:[8,24,25],At:20,For:[7,21],If:[8,12],In:[16,20,21,24,25],Is:21,It:[20,21],Its:[16,24,25],Not:21,Such:21,The:[2,3,5,7,8,9,12,16,20,21,24],There:21,These:[7,24],To:[9,20,21,26],Will:5,__init__:24,ab:12,abl:20,about:20,abov:7,absenc:7,absolut:[10,24],ac:12,accept:[16,22],account:12,accur:7,acm:25,across:[8,21],act:15,action:[2,3,4,5,7,8,9,15,16,20,21,25],action_dimens:2,action_metadata:7,action_scaling_coef:16,action_scaling_coeffici:[4,5],action_scaling_coefficienct:4,action_spac:[2,5,7,8,9,16],activ:[2,7,11,24],active_act:7,active_observ:[2,5,7],actual:21,adapt:20,add:12,add_to_buff:[2,5],addit:[8,16,21],advanc:[2,6,7,8,12],aec:9,affect:24,aforement:21,after:[4,9,12,20,21],afterward:16,agent:[0,7,8,9,15,17,19,20,21,24,25],agent_1:9,agent_2:9,agent_count:[15,24],agentid:9,aggreg:[24,25],ahead:[11,24],aicrowd:22,air:24,algorithm:[2,5,24,25],all:[2,7,8,9,16,20,21,24,26],allow:[20,24,25],alpha:4,alreadi:[20,21],also:[21,24,26],altern:[9,24],although:16,alwai:[7,12,21],am:3,among:20,amount:[12,21,24],an:[8,9,20,24,25],analyz:20,ani:[2,7,8,9,15,20,21,24],announc:22,annual:[8,21],anonym:20,anoth:20,ansi:9,anytim:20,apenergi:25,api:[2,9],appli:[4,8,25],applianc:[20,21],applic:[2,5],apply_act:7,approach:20,approxim:7,ar:[2,3,5,7,8,9,10,16,20,21,22,24],arg:[2,3,4,5],argument:[7,16,18],around:9,arrai:[9,11,24],arxiv:25,assign:8,associ:25,assum:[15,21],asynchron:16,attribut:9,autos:[7,12],autosize_cooling_devic:7,autosize_cooling_storag:7,autosize_dhw_devic:7,autosize_dhw_storag:7,autosize_electrical_storag:7,autosize_heating_devic:7,autosize_heating_storag:7,autosize_pv:7,auxiliari:8,avail:[2,5,7,20,21,24],available_nominal_pow:12,averag:[10,11,20,21,24],average_daily_peak:[10,21,24],average_unment_cooling_setpoint_differ:11,average_unmet_cooling_setpoint_differ:[11,24],avoid:7,backup:24,balanc:[4,12],base:[0,1,3,4,5,7,8,9,10,11,12,13,14,15,16,17,19,20,21],baselin:20,basic:20,basicbatteryrbc:3,basicrbc:3,batch:4,batch_siz:[4,16],batteri:[3,7,12,24],been:[20,24],befor:[10,20,24],begin:4,being:21,below:[7,16,20,24],benchmark:24,better:[20,21],between:[3,7,10,11,12,21,24],bia:16,board:20,bool:[5,7,8,12,16],boolen:18,both:20,bound:7,boundari:[2,5,7],box:[2,7,8],buffer:[2,4,5,16],build:[0,2,8,11,15,17,19,20,21,24,25],building_id:24,building_inform:[2,8],buildings_state_action_spac:[20,21],buildsi:25,bulb:[11,12,24],c:[11,12,13,24],calcul:[8,10,12,15,24],calendar:24,call:[2,6,8,9,16,20,24],can:[2,7,9,12,16,20,21,24,25],cant:[15,25],canva:14,canvas_s:14,capac:[3,4,7,8,12,16,20,21,24],capacity_histori:12,capacity_loss_coef:12,capacity_loss_coeffici:12,capacity_power_curv:12,capit:[24,25],carbon:[7,10,11,15,24],carbon_emiss:[10,15,24],carbon_intens:[7,11,24],carbonintens:[7,11],care:16,carnot:12,cast:16,categor:[2,5,13],categori:[13,24],cdoubl:16,central:[8,17,20,24],central_ag:8,certain:8,challeng:[8,21,22,24,25],chang:[21,24],channels_last:16,characterist:20,charg:[3,7,8,12,14,17,24],cheap:21,check:8,child:5,chill:24,citi:[20,24,25],citylearn:[20,21,22,26],citylearn_env:[8,17],citylearn_pettingzoo:[0,19],citylearnenv:[8,17],citylearnpettingzooenv:9,classic:9,climat:20,clip:10,close:9,code:20,coeffici:[4,12],cold:7,collect:24,color:14,combin:24,come:21,common:[8,24],commun:20,compar:[7,25],competit:22,complet:[7,8],complex128:16,complex:16,compris:20,comput:[16,24,25],condit:[7,20],confer:25,connect:[9,15,25],consecut:[10,12,24],consid:12,consist:20,constant:21,consum:[12,20,21,24,25],consumpt:[7,10,12,15,20,21,24],contain:[8,16,20],control:[3,7,8,15,20,21,24,25],convert:[9,16,18],cool:[7,11,12,20,21,24],cooling_by_devic:21,cooling_cop:12,cooling_demand:[7,8,11,12],cooling_devic:7,cooling_electricity_consumpt:[7,8],cooling_stoag:7,cooling_storag:[7,24],cooling_storage_act:7,cooling_storage_electricity_consumpt:[7,8],cooling_storage_soc:[21,24],coordin:[20,24,25],cop:12,copi:11,core:8,correct:21,correl:[8,20],cost:[7,10,20,21,25],cost_funct:[0,19],cost_rbc:21,costfunct:10,could:[20,21],count:9,cpu:[16,20],creat:21,csv:24,cuda:16,current:[2,3,5,6,7,8,12,21,24],curv:[21,24,25],custom:[2,6,8,24],custom_modul:24,customreward:24,cycl:12,cyclic:[2,5],dai:[10,11,24],daili:[10,20,21,24],daily_time_step:10,data:[0,7,8,9,19,20],dataset:[11,20],date:20,day_typ:[11,24],daylight:[11,13,24],daylight_savings_statu:[11,24],debug:8,decai:4,decemb:24,decent:21,decentr:20,decid:[21,24],decis:20,decreas:21,deep:25,def:24,defin:[2,5,8,12,16,20,24],definit:24,degrad:12,dei:25,delight:22,demand:[7,8,10,11,12,20,21,24,25],depend:[12,21,26],deriv:12,describ:24,descript:24,design:[3,20],desir:16,destination_directori:11,detail:[20,24],determin:21,determinist:4,deterministic_start_time_step:4,develop:9,devic:[2,5,7,8,12,16,21,24],dhw:[7,20,21],dhw_demand:[7,8,11],dhw_devic:7,dhw_electricity_consumpt:[7,8],dhw_stoag:7,dhw_storag:[7,24],dhw_storage_act:7,dhw_storage_electricity_consumpt:[7,8],dhw_storage_soc:24,diagnost:8,dict:[7,8,9,18],dictionari:[8,18],differ:[8,10,11,12,20,21,24,25],diffus:[11,24],diffuse_solar_irradi:[11,24],diffuse_solar_irradiance_predicted_12h:[11,24],diffuse_solar_irradiance_predicted_24h:[11,24],diffuse_solar_irradiance_predicted_6h:[11,24],dimens:4,dioxid:7,direct:[11,24],direct_solar_irradi:[11,24],direct_solar_irradiance_predicted_12h:[11,24],direct_solar_irradiance_predicted_24h:[11,24],direct_solar_irradiance_predicted_6h:[11,24],discharg:[3,7,8,12,17,24],discount:4,displai:9,distribut:[20,24,25],district:[20,21,24,25],divers:24,doc:24,document:[9,18],doe:[2,21],doi:25,domest:[7,11,24],don:21,done:[5,8,9,16],dot:15,doubl:16,draw_build:14,draw_lin:14,draw_obj:14,dry:[11,12,24],dtype:16,due:24,dump:18,dure:[2,5,7,8,12,24],e:[7,8,10,11,13,15,16,24,25],e_0:15,e_:10,e_i:10,e_n:15,each:[7,8,9,12,15,20,21,24,25],easi:[24,25],easili:[21,25],edu:20,effect:7,effici:[12,25],efficiency_histori:12,efficiency_sc:12,either:[20,21],electr:[3,7,10,11,12,15,20,21,24,25],electric_consumption_appli:21,electric_consumption_dhw:21,electric_gener:21,electrical_consumption_dhw_storag:21,electrical_consumption_h:21,electrical_devic:7,electrical_storag:[7,24],electrical_storage_act:7,electrical_storage_electricity_consumpt:[7,8],electrical_storage_soc:24,electricdevic:12,electricheat:[7,12],electricity_consumpt:[12,15,24],electricity_pr:[11,24],electricity_pric:[15,24],electricity_pricing_predicted_12h:[11,24],electricity_pricing_predicted_24h:[11,24],electricity_pricing_predicted_6h:[11,24],electrifi:24,element:10,els:[5,8,12],elsewher:9,email:20,emand:12,emiss:[7,10,11,15,24],emit:8,emmiss:7,empti:21,enabl:24,encod:[2,4,5,13],encourag:20,end:[4,5,7,8,24],end_exploration_time_step:[4,5],energi:[7,8,10,12,14,17,20,21,24,25],energy_bal:12,energy_from_cooling_devic:[7,8],energy_from_cooling_device_to_cooling_storag:[7,8],energy_from_cooling_storag:[7,8],energy_from_dhw_devic:[7,8],energy_from_dhw_device_to_dhw_storag:[7,8],energy_from_dhw_storag:[7,8],energy_from_electrical_storag:[7,8],energy_from_heating_devic:[7,8],energy_from_heating_device_to_heating_storag:[7,8],energy_from_heating_storag:[7,8],energy_model:[0,7,19],energy_simul:[7,11],energy_to_electrical_storag:[7,8],energysimul:[7,11],engin:21,enough:20,enter:8,entir:[20,24],env:[8,9,21],environ:[2,6,7,8,9,10,12,17,20,21,25],episod:[5,8,17,20,21],epsilon:16,equal:20,equip:[11,24],error:8,estim:7,estimate_action_spac:7,estimate_observation_spac:7,evalu:[8,10,20,24,25],event:12,ever:7,everi:[3,16,24,25],exampl:[7,10,13,16,20,21,24],exceed:8,except:8,exclus:21,expect:8,exploit:4,explor:[4,5,20],expon:12,extent:21,facilit:[24,25],factor:[4,10,12,20,21],fals:[8,16],featur:[7,21],feel:21,file:[18,20,21,24],filepath:[8,18],find:[7,9,10],first:[8,26],flat:[21,24],flatten:[21,24,25],flexibl:[7,10,24],float16:16,float64:16,folder:20,follow:[8,24],forecast:7,format:[2,16],former:16,forward:16,four:20,frac:12,fraction:[7,8,12],frame:9,free:21,from:[2,5,7,9,11,12,20,21,24],from_parallel:9,full:[9,21,24],further:8,futur:21,g:[11,13,16,24,25],gener:[4,7,12,20,21,24,25],get:[2,5,7,8,12],get_background:14,get_building_inform:[8,20],get_coord:14,get_cop:12,get_current_effici:12,get_default_shared_observ:8,get_encoded_observ:5,get_exploration_act:5,get_gener:12,get_info:8,get_input_pow:12,get_max_input_pow:12,get_max_output_pow:12,get_nam:11,get_normalized_observ:5,get_normalized_reward:5,get_plot:14,get_post_exploration_act:5,get_reward:8,get_schema:11,github:20,given:[2,5,7,12,16,21,24],go:21,good:21,gpu1:16,gpu:20,graphic:9,gregor:[15,25],grid:[7,11,12,15,21,24,25],group:24,guarante:24,gym:[2,7,8,24,25],ha:[5,8,10,21],had:20,half:16,hand:[9,20],happen:21,have:[8,20,21,24,25],heat:[7,11,12,21,24],heater:[21,24],heating_by_devic:21,heating_cop:12,heating_demand:[7,8,11,12],heating_devic:7,heating_electricity_consumpt:[7,8],heating_stoag:7,heating_storag:[7,24],heating_storage_act:7,heating_storage_electricity_consumpt:[7,8],heating_storage_soc:24,heatpump:[7,12],help:[8,24,25],henc:7,henz:[15,25],here:[9,20],hidden:4,hidden_dim:16,hidden_dimens:4,hidden_s:16,high:[7,24,25],higher:21,histori:2,holidai:11,hook:16,host:[16,22],hot:[7,11,24],hour:[3,11,21,24],hourli:[12,20],how:[21,24],howev:21,human:9,humid:[11,21,24],i:[7,8,10,21,24,25],id:6,identity_matrix:13,ignor:16,implement:[2,6,9,21,24,25],in_featur:16,inact:[7,11,24],incentiv:21,includ:[2,5,8,21,24],increas:21,independ:15,independentsacreward:15,index:[5,14,25],indic:[5,7,11,24],indoor:7,indoor_dry_bulb_temperatur:[11,24],indoor_relative_humid:[11,24],info:[8,9],inform:[8,15,20,22],infrequ:18,inherit:24,init_w:16,initi:[2,6,7,8,9,12,13,24],initial_soc:12,input:12,input_pow:12,instanc:[8,16],instead:[16,21],integr:16,interact:[15,25],intern:[24,25],internal_observation_count:5,invalid:8,invert:[11,12],inverter_ac_power_per_kw:12,inverter_ac_power_per_w:12,inverter_ac_power_perk_w:12,irradi:[11,21,24],irrespect:8,is_paralleliz:9,issu:9,item_1:9,item_2:9,iter:[11,12,15,25],its:[3,7,12,21,24],j:[16,25],januari:[20,24],json:[8,18,20,21,24],just:9,keep:2,kei:8,kept:[2,9],keyword:[7,16,18],kg_co2:[7,8,11,15],kgco2:24,known:[2,5],kw:[11,12],kwarg:[2,3,4,5,7,8,12,15,18],kwh:[7,8,11,12,15,24],kwhcapac:24,last:24,later:21,latest:12,latter:16,leader:20,leaderboard:20,learn:[4,7,8,15,20,24,25],least:21,length:[8,10],librari:20,like:9,limit:[7,8,12,14,21],line_color:14,linear:16,link:20,linux:20,list:[2,3,4,5,7,8,9,10,11,12,13,15,24],load:[7,10,11,15,18,20,21,24,25],load_ag:8,load_factor:[10,21,24],log:8,log_std_max:16,log_std_min:16,longer:9,look:9,loss:12,loss_coeffici:12,losss:12,lost:12,low:[7,21],lower:7,lr:4,lvert:10,m2:24,m:11,machineri:25,made:20,mai:[7,8,10,21,24],main:[9,20,24],make:[7,21,24],make_citylearn_env:9,manag:[8,17,25],mani:[8,24],map:[2,7,8,15],marl:15,marlisa:[15,25],max:[12,15],max_electric_pow:12,max_input_pow:12,max_output_pow:12,maximum:[2,3,5,7,12,20,21],maximum_demand:7,mayb:8,mean:[10,20,24],measur:24,meet:[7,12,24],memori:16,memory_format:16,mention:21,messag:8,met:7,metadata:[2,9],meth:8,method:[9,16,20],metric:[20,21,24],midnight:3,min:12,minim:[20,21],minimum:[2,5,7,10,12,21],minu:21,mode:[9,12],model:[21,24,25],modifi:[16,20,21],modul:[0,1,19,25],mondai:[11,24],month:[11,24],more:[7,15,22],move:16,much:24,multi:[15,17,20,24,25],multipl:20,multipli:5,must:[7,9,12,20,24],n:[10,15],nagi:[15,25],name:[2,7,8,9,11,24],nan:10,ndarrai:[7,8,12],necessari:20,need:[7,9,12,16,21],neg:[7,21],net:[7,10,20,21,24],net_electric_consumpt:21,net_electricity_consumpt:[7,8,10,24],net_electricity_consumption_emiss:[7,8],net_electricity_consumption_pric:[7,8],net_electricity_consumption_without_storag:[7,8],net_electricity_consumption_without_storage_and_pv:[7,8],net_electricity_consumption_without_storage_and_pv_emiss:[7,8],net_electricity_consumption_without_storage_and_pv_pric:[7,8],net_electricity_consumption_without_storage_emiss:[7,8],net_electricity_consumption_without_storage_pric:[7,8],network:[9,24,25],neurip:22,next:[2,6,7,8,12,21],next_observ:5,next_stat:16,next_time_step:[2,6,7,8,12],nine:20,nn:16,nomin:12,nominal_pow:[7,12],nominal_pw:7,non:[2,5,11,20,21,24],non_block:16,non_shiftable_load:[11,24],non_shiftable_load_demand:[7,8],none:[4,5,6,7,8,9,10,11,12,15,16,17],nonorm:13,nor:8,normal:[2,5,7,13,20,21],note:[2,3,6,7,8,10,12,15,24],noth:2,novemb:25,np:11,num_act:16,num_build:14,num_input:16,num_mov:9,number:[2,4,6,8,9,10,15,17],numer:11,numpi:[7,8,9,12],ny:25,obeserv:[2,5,7],object:[6,7,8,10,11,13,14,15,16,17,21,24,25],observ:[2,3,4,5,7,8,9,13,21],observation_dimens:4,observation_metadata:7,observation_nam:[2,8],observation_spac:[2,7,8,9],obtain:[20,21],od:7,offset:[7,24],often:9,one:[16,20,21],onehotencod:[2,5,13],ones:16,onli:[8,16,20,21,24],open:25,openai:[24,25],oper:[18,24,25],optim:3,optimizedrbc:3,option:[4,5,6,7,10,11,12,15,17],order:[2,5,8,21,24,25],os:20,other:[3,7,8,9,15,18,20,21,25],otherwis:12,our:20,out_featur:16,outdoor:[7,11,12,21,24],outdoor_dry_bulb_temperatur:[11,12,24],outdoor_dry_bulb_temperature_predicted_12h:[11,24],outdoor_dry_bulb_temperature_predicted_24h:[11,24],outdoor_dry_bulb_temperature_predicted_6h:[11,24],outdoor_relative_humid:[11,24],outdoor_relative_humidity_predicted_12h:[11,24],outdoor_relative_humidity_predicted_24h:[11,24],outdoor_relative_humidity_predicted_6h:[11,24],output:[7,11,12],output_pow:12,outsid:9,over:[10,20,24],overal:[21,24,25],overrid:[2,6,8],overridden:16,overs:12,own:[20,24],packag:19,page:[22,25],pair:8,parallelenv:9,paramet:[5,7,8,10,12,13,16,18,20],pars:[7,18],part:24,parti:26,particip:20,particular:20,pass:16,past:20,path:[8,11,24],pathlib:[8,11],pathnam:18,peak:[10,20,21,24],peak_demand:[10,21,24],per:[4,11,12],perform:[10,12,16,20,24],period:[10,20,24,25],periodicnorm:[2,5,13],pettingzoo:9,phase:[4,20],photovolta:24,physic:8,pin:16,pip:26,place:16,plai:[9,20],pleas:[20,22],plug:[7,11,20,24],pm:3,point:16,polici:[5,20],policynetwork:16,posit:7,possess:24,possibl:[7,16,21],possible_ag:9,post:5,potenti:20,power:[7,10,12,20,21,24,25],power_efficiency_curv:12,pp:25,pre:[20,24],predict:[11,21,24],predictor:21,preprocess:[0,2,5,19],previou:5,price:[7,10,11,15,21,24,25],print:9,priorit:24,proceed:25,process:[7,21],product:12,profil:[20,24],properti:[2,4,5,6,7,8,12,15,17,24],proport:21,provid:[2,3,5,7,20,21,24],pseudorandom:4,pump:[12,21,24],push:16,pv:[7,8,11,12,24],py:[2,5,20],python:20,pytorch:20,quadrat:[10,24],quantifi:[12,24],quantiti:12,quotient:12,r:25,radiat:21,rais:[8,10,24,25],ramp:[10,20,21,24],randomli:[2,5],rang:[11,24],rate:[4,7,11,24],ratio:[10,24],raw_env:9,rbc:[0,1,5,20],reach:[8,17],read:20,read_glow_imag:14,read_imag:14,read_json:18,real:21,reason:8,receiv:20,recip:16,recommend:15,reduc:[24,25],reduct:21,refer:[15,24],reflect:21,regard:8,regardless:[21,24],regist:16,regressionbuff:16,reinforc:[15,20,24,25],rel:[11,21,24],relat:21,releas:[9,12,21,24],removefeatur:[2,5,13],render:[0,8,9,19],render_mod:9,renderbuild:14,repeat:4,replai:[2,4,5],replay_buffer_capac:4,replaybuff:16,repositori:20,repres:21,represent:8,research:25,reserv:11,reset:[2,6,7,8,9,12],reset_time_step:[2,6],reshap:[24,25],respect:[2,5,7,11,12,16],respons:[20,24,25],rest:20,result:[4,7,8,21],review:25,reward:[4,5,8,9,15,16,20,21],reward_funct:[0,8,19,20,24],reward_sc:4,rewardfunct:[8,15,24],rgb_arrai:9,rl:[0,7,19,20,24,25],rl_agent:20,rl_cost:20,rlc:[0,1,5],roll:[10,24],rough:7,rule:21,run:[16,17,20,21],runtim:24,rvert:10,s:[3,8,16,17,20,21,24,25],sac:[0,1,15],sacbasicbatteryrbc:5,sacbasicrbc:5,sacoptimizedrbc:5,sacrbc:5,safety_factor:12,same:[8,9,20,24],sampl:[2,5,16,20],satisfi:24,save:[11,13,24],scale:[4,7,12],schema:[8,9,24],score:20,search:25,second:6,seconds_per_time_step:6,see:[15,16,20,21],seed:[4,9],seem:21,select:[5,7,15,20,21,25],select_act:[2,3,5],self:[7,16,24],sequenti:[15,20,25],seri:[2,7,8,10,11,12],set:[6,7,8,9,12,16,20,24,25],set_encod:[2,5],set_energy_bal:12,set_network:5,setpoint:[11,24],setup:[8,24],shape:[15,20,21,25],share:[15,20],shared_observ:8,shiftabl:[11,20,21,24],should:[7,8,9,16,21],sign:20,signal:[8,11],signatur:16,silent:16,similar:[2,16],simpl:20,simplejson:18,simpli:21,simul:[0,2,7,8,19,20,24],simultan:20,sinc:16,singl:[12,20,24],size:4,smooth:21,smoothen:[24,25],so:[9,21],soc:[12,21,24],soc_init:12,softqnetwork:16,solar:[7,11,12,20,21,24],solar_gen:21,solar_gener:[7,8,11,24],solv:8,some:[20,21],sourc:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,24,25],space:[2,7,8,9,11,24],special:11,specif:[9,24],specifi:[10,12,24],speed:7,squar:21,stabl:7,stage:20,stageofchalleng:20,standard:[24,25],standbi:12,start_training_time_step:4,state:[2,5,6,7,8,12,16,17,20,21,24],statu:11,step:[2,3,4,5,6,7,8,9,10,12,21,24],still:20,stop:4,storag:[2,3,5,7,8,12,17,21,24,25],storagedevic:12,storagetank:[7,12],store:[7,12,21,24],str:[2,6,7,8,9,11,13,18],string:9,sub:8,subclass:[2,6,16],subject:20,sublist:8,submit:20,submodul:19,subpackag:19,subprocess:9,success:18,successfulli:8,sum:[8,10,21,24],sum_:10,summari:8,sundai:[11,24],suppli:[7,12,21,24],support:9,system:[2,3,5,7,11,25],t:[7,13,21],t_target_cool:12,t_target_h:12,take:[4,9,16,20,21],tank:24,target:[12,16],target_cooling_temperatur:12,target_heating_temperatur:12,task:8,tau:4,team_nam:20,technic:12,techniqu:25,temperatur:[4,11,12,21,24],tempor:7,tend:20,tensor:16,tensorflow:20,term:4,termin:17,textrm:[10,12,15],th:10,than:[7,20,21],thei:21,them:[7,16,21],therefor:[20,21],thermal:21,theses:21,thi:[2,7,16,20,21],think:21,third:26,those:21,three:[20,24],through:[15,21,24],thu:12,time:[2,3,4,5,6,7,8,10,11,12,17,21,24],time_step:[2,5,6,7,8,12],timelimit:8,timelin:20,torch:16,total:[11,12,20,24],track:22,tradit:17,train:[7,16,20],transform:[2,5,13],transmiss:[24,25],transport:25,tri:16,tune:20,tupl:8,type:[2,3,5,7,8,10,11,12,16,18,20,24],uid:6,unchang:16,undefin:8,under:[20,24],underli:8,understand:21,union:[7,8,11,12,13],uniqu:[6,7],unit:[12,24],unknownschemaerror:8,unord:13,until:17,up:[7,9,20,24],updat:[2,4,5,12,20,24],update_cool:7,update_dhw:7,update_electrical_storag:7,update_electricity_consumpt:12,update_h:7,update_per_time_step:4,update_vari:[7,8],upper:7,urban:25,us:[2,5,7,8,9,10,12,13,15,18,20,21,24,26],usa:25,usag:25,user:[9,24],utexa:20,util:[0,9,19,24],v1:25,valid:2,valu:[2,3,5,6,7,8,9,10,11,12,14,15,18,21],variabl:[7,9,16,20,21],vazquez:25,virtual:[21,24],visit:22,volum:[11,24],w:[11,12,24],wa:[8,20],want:21,water:[7,11,24],we:[20,21,22],weather:[7,11,20,24],week:[11,24],weight:[11,16,20,24],well:[20,24],were:20,when:[2,6,8,12,16,21],where:[8,9,10,11,15,20],wherea:24,whether:21,which:[7,8,9,20,21,24,25],whose:16,why:21,window:[10,20],wish:21,within:[16,20,21],without:[9,15],world:21,worth:21,would:21,wrap:9,wrapper:9,write:18,write_json:18,written:20,x_max:13,x_min:13,ye:21,year:[20,21,24],york:25,you:[9,21],your:21,z:25,zoltan:25,zone:[11,20,24]},titles:["citylearn package","citylearn.agents package","citylearn.agents.base module","citylearn.agents.rbc module","citylearn.agents.rlc module","citylearn.agents.sac module","citylearn.base module","citylearn.building module","citylearn.citylearn module","citylearn.citylearn_pettingzoo module","citylearn.cost_function module","citylearn.data module","citylearn.energy_model module","citylearn.preprocessing module","citylearn.rendering module","citylearn.reward_function module","citylearn.rl module","citylearn.simulator module","citylearn.utilities module","citylearn","2020","2021","2022","CityLearn Challenge","CityLearn Environment","CityLearn","Usage"],titleterms:{"2020":20,"2021":21,"2022":22,"function":24,"public":25,action:24,agent:[1,2,3,4,5],base:[2,6],build:7,challeng:[20,23],cite:25,citylearn:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,25],citylearn_pettingzoo:9,content:25,cost:24,cost_funct:10,data:11,deadlin:20,descript:25,energy_model:12,environ:24,faq:21,indic:25,instal:26,instruct:20,member:20,modul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],object:20,observ:24,packag:[0,1],preprocess:13,rbc:3,relat:25,render:14,reward:24,reward_funct:15,rl:16,rlc:4,rule:20,sac:5,simul:17,submiss:20,submodul:[0,1],subpackag:0,tabl:25,team:20,usag:26,util:18}})