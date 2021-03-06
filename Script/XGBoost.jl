using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))

using  DataFrames, CSV ,MLJ, MLJLinearModels, Random,MLJXGBoostInterface
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


xgb = XGBoostClassifier()


m2 = fit!(machine(TunedModel(model = xgb,
                            resampling = CV(nfolds = 5),
                            tuning = Grid(goal = 20),
                            range = [range(xgb, :eta,
                                           lower = 1e-2, upper = .1, scale = :log),
                                           range(xgb, :num_round, lower = 50, upper = 500),
                                     range(xgb, :max_depth, lower = 2, upper = 7)],measure = auc),
select(train_data_nv,Not(:precipitation_nextday)),train_data_nv.precipitation_nextday),verbosity = 2 )
report(m2)
report(m2).best_model
