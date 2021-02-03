#creates a temporal hierarchy of a count time series and reconciles it via prob prog.
library(tscount)
library(forecast)
#managing hierarchies
library(hts)
#managing temporal hierarchies
library(thief)

buildMatrix <- function() {
  A <- matrix(data = 0, nrow = length(upperIdx), ncol = length(bottomIdx))
  maxFreq <- frequency(trainHier[[1]])
  counter <- 1
  for (ii in (2:length(trainHier))){
    currentFreq <- frequency(trainHier[[ii]])
    aggregatedTs <- currentFreq
    howManyBottomToSum <- maxFreq / currentFreq
    offset <- 1
    for (jj in (1:aggregatedTs)) {
      A[counter, offset : (offset + howManyBottomToSum - 1)] <- 1
      offset <- offset + howManyBottomToSum
      counter <- counter + 1
    }
  }
  return (t(A))
}

#weekly time series with low counts from thief
autoplot(AEdemand[,13])
ts <- AEdemand[,13]
#cut the  last year of data 
train <- window(ts, end = max(time(ts))-1)
trainHier <- tsaggregates(train, align = "end")
autoplot(trainHier)

test <-  window(ts, start = 2015)

testHier <-  window(ts, start = 2015)

#campy is recorded every 28 days, difficult to manage into a hierarchy.
#taken from the vignette.
# interventions <- interv_covariate(n = length(campy), tau = c(84, 100), delta = c(1, 0))
# campyfit_pois <- tsglm(campy, model = list(past_obs = 1, past_mean = 13), xreg = interventions, distr = "poisson")
# campyfit_nbin <- tsglm(campy, model = list(past_obs = 1, past_mean = 13), xreg = interventions, distr = "nbinom")
# 
# #example of scoring rules
# rbind(Poisson = scoring(campyfit_pois), NegBin = scoring(campyfit_nbin))
timeseries <- Seatbelts[, "VanKilled"]
regressors <- cbind(PetrolPrice = Seatbelts[, c("PetrolPrice")],linearTrend = seq(along = timeseries)/12)
train <- window(timeseries, end = 1981 + 11/12)
regressors_train <- window(regressors, end = 1981 + 11/12)
fit_pois <- tsglm(train, model = list(past_obs = c(1, 12)), link = "log", distr = "poisson", xreg = regressors_train)

test <- window(timeseries, start = 1982)
regressors_test <- window(regressors, start=1982)

#fit_negbin <- tsglm(timeseries_until1981, model = list(past_obs = c(1, 12)), link = "log", distr = "nbinom", xreg = regressors_until1981)# #example of scoring rules
#scoring
scoring(fit_pois)


