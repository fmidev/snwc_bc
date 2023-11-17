# 1) convert vfld files to sqlite files to tmp folder (yyyy/mm/all the days saved in by hour)
# 2) merge all sqlite files created to 0ne file/month (narrow tables)
# 3) remove all the files from tmp folder
# 4) need to reduce data --> therefore only every second forecast origtime and every second forecast leadtime is used

library(harpIO)

args = commandArgs(trailingOnly=TRUE)
first_fcst = args[1] 
last_fcst = args[2] 
base = args[3]

param <- c("D10m","S10m","T2m","RH2m","Pmsl","Q2m","CClow","Gmax")

tmp <- as.character(first_fcst)
yr <- substr(tmp,1,4) # year

#/data/MNWC/ecfadm_vfldMNWC_preop/2019/01/01
vfld_path       <- paste(base,"/MNWC",yr,sep="")
## kommentoi pois
#first_fcst      <- 202006010100
#last_fcst       <- 202006030100
#                    202006010100
###
fcst_freq       <- "4h"
fcst_models     <- c("MNWC_preop")
fcst_lead_times <- c(0,1,2,4,6,8)
tmp_folder      <- paste(base,"/tmp",sep="")

read_forecast(
  start_date=first_fcst ,
  end_date=last_fcst,
  fcst_model=fcst_models,
  parameter=param,
  lead_time = fcst_lead_times,
  members=NULL,
  by = fcst_freq,
  file_path = vfld_path,
  file_format = "vfld",
  file_template = "vfld",
  #vertical_coordinate = NULL, #c("pressure", "model", "height", NA),
  #clim_file = NULL,
  #clim_format = file_format,
  #interpolation_method = "nearest",
  #use_mask = FALSE,
  #sqlite_path = "/data/hietal/testi",
  #sqlite_template = "fctable_det",
  transformation_opts = interpolate_opts(
    correct_t2m    = FALSE,
    keep_model_t2m = TRUE
  ),
  output_file_opts= sqlite_opts(
    path = tmp_folder,
    template = "fctable_det",
    index_cols = c("fcdate", "leadtime", "SID"),
    remove_model_elev = FALSE), 
  
  #sqlite_synchronous = c("off", "normal", "full", "extra"),
  #sqlite_journal_mode = c("delete", "truncate", "persist", "memory", "wal", "off"),
  return_data = FALSE,
  show_progress = FALSE,
  stop_on_fail = FALSE
)


#fcst_freq       <- "4h"
#fcst_models     <- c("MNWC_preop")
fcst_lead_times <- c(0,1,3,5,7,9)
#tmp_folder      <- "/tmp"
tmp <- as.numeric(first_fcst)+200
next_fcst <- as.character(tmp)

read_forecast(
  start_date=next_fcst ,
  end_date=last_fcst,
  fcst_model=fcst_models,
  parameter=param,
  lead_time = fcst_lead_times,
  members=NULL,
  by = fcst_freq,
  file_path = vfld_path,
  file_format = "vfld",
  file_template = "vfld",
  #vertical_coordinate = NULL, #c("pressure", "model", "height", NA),
  #clim_file = NULL,
  #clim_format = file_format,
  #interpolation_method = "nearest",
  #use_mask = FALSE,
  #sqlite_path = "/data/hietal/testi",
  #sqlite_template = "fctable_det",
  transformation_opts = interpolate_opts(
    correct_t2m    = FALSE,
    keep_model_t2m = TRUE
  ),
  output_file_opts= sqlite_opts(
    path = tmp_folder,
    template = "fctable_det",
    index_cols = c("fcdate", "leadtime", "SID"),
    remove_model_elev = FALSE), 
  
  #sqlite_synchronous = c("off", "normal", "full", "extra"),
  #sqlite_journal_mode = c("delete", "truncate", "persist", "memory", "wal", "off"),
  return_data = FALSE,
  show_progress = FALSE,
  stop_on_fail = FALSE
)







