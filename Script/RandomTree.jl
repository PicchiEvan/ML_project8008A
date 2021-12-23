using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using DataFrames, CSV ,MLJ, MLJLinearModels, Random,MLJDecisionTreeInterface

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


#Model with standardization

model_stand = @pipeline(Standardizer(features =[:ALT_sunshine_4,:ZER_sunshine_1,:ABO_sunshine_4,:CHU_sunshine_4,:DAV_sunshine_4,:SAM_sunshine_4,:ZER_sunshine_4], ignore = true),RandomForestClassifier(rng = 1))
    self_tuning_model_stand = TunedModel(model = model_stand,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model_stand, :(random_forest_classifier.n_trees), values = [100,150,200,250,300,350,400,450,500,550,600]),
                                   measure = auc)
    self_tuning_mach_stand =fit!( machine(self_tuning_model_stand,
                               select(train_data_nv, Not(:precipitation_nextday)),
                               train_data_nv.precipitation_nextday),verbosity = 2 )



report(self_tuning_mach_stand).best_model

#Model_without standardization
#=model = RandomForestClassifier(rng = 1)
    self_tuning_model = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model, :n_trees, values = [100,150,200,250,300,350,400,450,500,550,600]),
                                   measure = auc)
    self_tuning_mach =fit!( machine(self_tuning_model,
                               select(train_data_nv, Not(:precipitation_nextday)),
                               train_data_nv.precipitation_nextday),verbosity = 2 )



report(self_tuning_mach).best_model =#
