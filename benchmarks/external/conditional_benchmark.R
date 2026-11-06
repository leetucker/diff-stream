s<-jsonlite::fromJSON(commandArgs(TRUE)[1])
if(!requireNamespace('did',quietly=TRUE))stop('did is missing')
peak<-function() {
 z<-readLines('/proc/self/status');as.numeric(strsplit(trimws(sub('VmHWM:','',grep('^VmHWM:',z,value=TRUE))),' +')[[1]][1])/1024
}
baseline<-peak();start<-proc.time()[[3]]
d<-read.csv(s$data)
a<-did::att_gt(yname='y',tname='time',idname='id',gname='first_treat',data=d,xformla=~x+z,
               est_method=s$method,bstrap=FALSE,cband=FALSE)
cat('RESULT ',jsonlite::toJSON(list(wall_s=unname(proc.time()[[3]]-start),peak_memory_mb=peak(),
    baseline_memory_mb=baseline,peak_disk_mb=0,checksum=sum(a$att),n_obs=nrow(d)),auto_unbox=TRUE),'\n',sep='')
