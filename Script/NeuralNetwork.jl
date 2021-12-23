
using Pkg; Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using  DataFrames, CSV,MLJ, MLJLinearModels, Random,MLJFlux, Flux

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




#modelNN = NeuralNetworkClassifier(builder = MLJFlux.Short(σ = sigmoid),optimiser = ADAM(),batch_size = 128) (without standardizer but for neural network it is better with)
modelNN = @pipeline(Standardizer(features =[:ALT_sunshine_4,:ZER_sunshine_1,:ABO_sunshine_4,:CHU_sunshine_4,:DAV_sunshine_4,:SAM_sunshine_4,:ZER_sunshine_4], ignore = true),
                   NeuralNetworkClassifier(
                         builder = MLJFlux.Short(σ = sigmoid),optimiser = ADAM(),batch_size = 128,rng = 1))

tuned_modelNN = TunedModel(model = modelNN,
                          resampling = CV(nfolds = 5),
                          range = [range(modelNN,:(neural_network_classifier.builder.dropout),values = [0., .1, .2]),
                                    range(modelNN,:(neural_network_classifier.builder.n_hidden),values = [100,200,300]),
                                   range(modelNN, :(neural_network_classifier.epochs),values = [50, 100, 150])])
                                   #,measure = Flux.crossentropy

mach_final_nn = fit!(machine(tuned_modelNN,
                     select(train_data_nv,Not(:precipitation_nextday)),train_data_nv.precipitation_nextday),verbosity = 2)
report(mach_final_nn).best_model
