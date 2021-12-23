using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using  DataFrames, CSV ,MLJ, MLJLinearModels, Random

function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end
train_data_nv = DataFrame(CSV.read(joinpath(@__DIR__,  "..","data","train_data_nv.csv"), DataFrame))
test_data_nv = DataFrame(CSV.read(joinpath(@__DIR__,  "..","data",  "test_data_nv.csv"), DataFrame))
coerce!(train_data_nv, :precipitation_nextday => Binary)
coerce!(test_data_nv, :precipitation_nextday => Binary)

dropmissing!(train_data_nv)
dropmissing!(test_data_nv)





function tune_model(model, data;lower = 1e-3, upper = 1e7, goal=100,stand = false)
	vari = :lambda
	if(stand==true)
		vari =  :(logistic_classifier.lambda)
	end

	tuned_model = TunedModel(model = model,
	                         resampling = CV(nfolds = 10),
	                         tuning = Grid(goal = goal),
	                         range = range(model,vari,
									       scale = :log,
									       lower = lower, upper = upper),
	                         measure = auc)
	fit!(machine(tuned_model, select(data, Not(:precipitation_nextday)), data.precipitation_nextday),verbosity = 2 )
end

# Uncomment the auto tuning model you want to test

#1) : L1, no standardization (no convergence)
#=mach_log_L1 = tune_model(LogisticClassifier(penalty=:l1),train_data_nv)
report(mach_log_L1).best_model
report(mach_log_L1)=#

#2) : L2, no standardization
#=mach_log_L2 = tune_model(LogisticClassifier(penalty=:l2),train_data_nv)
report(mach_log_L2).best_model=#
#3): L1, with standardization (not converge)
#= mach_log_L1_norm = tune_model(@pipeline(Standardizer(features =[:ZER_sunshine_1,:ALT_sunshine_4], ignore = true),LogisticClassifier(penalty=:l1)),train_data_nv;stand=true)
report(mach_log_L1_norm).best_model=#
#4) : L2, with standardization
#=mach_log_L2_norm = tune_model(@pipeline(Standardizer(features =[:ALT_sunshine_4,:ZER_sunshine_1,:ABO_sunshine_4,:CHU_sunshine_4,:DAV_sunshine_4,:SAM_sunshine_4,:ZER_sunshine_4], ignore = true),LogisticClassifier(penalty=:l2)),train_data_nv;stand=true)
report(mach_log_L2_norm).best_model=#
