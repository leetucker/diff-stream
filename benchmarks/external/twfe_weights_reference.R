# Native TwoWayFEWeights references. No replacement of package functions.
args<-commandArgs(TRUE);out<-args[1]
library(TwoWayFEWeights)
fixest::setFixest_nthreads(2)
fixest::setFixest_estimation(fixef.tol=1e-10,fixef.rm='none')
configs<-read.csv(file.path(out,'configs.csv'),na.strings='')
rows<-list()
for(i in seq_len(nrow(configs))) {
 c<-configs[i,];d<-read.csv(file.path(out,paste0(c$case,'.input.csv')))
 if(c$mode=='complete') d<-d[d$id %in% names(which(table(d$id)==length(unique(d$time)))),]
 controls<-if(c$controlled) c('x','z') else NULL
 w<-if(is.na(c$weight)) NULL else d[[c$weight]]
 error<-''
 r<-tryCatch(twowayfeweights(d,'y','id','time','d',type='feTR',controls=controls,weights=w,summary_measures=TRUE),error=function(e){error<<-conditionMessage(e);NULL})
 if(!is.null(r)) write.csv(r$dat_result,file.path(out,paste0(c$name,'.csv')),row.names=FALSE)
 value<-function(n) if(is.null(r)||is.null(r[[n]])) NA_real_ else r[[n]]
 rows[[i]]<-data.frame(name=c$name,status=if(is.null(r)) 'error' else 'ok',error=error,
                      beta=value('beta'),nr_plus=value('nr_plus'),nr_minus=value('nr_minus'),
                      sum_plus=value('sum_plus'),sum_minus=value('sum_minus'),
                      sensibility=value('sensibility'),sensibility2=value('sensibility2'))
}
write.csv(do.call(rbind,rows),file.path(out,'models.csv'),row.names=FALSE)
writeLines(c('Native TwoWayFEWeights feTR; fixest fixef.rm=none, fixef.tol=1e-10.',
 '2.1.0 requires weights as a numeric vector despite its column-name documentation.',capture.output(sessionInfo())),file.path(out,'R_session.txt'))
