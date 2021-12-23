### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ f2bfbb40-5f40-11ec-18f2-d12eb2303d46
begin
using Pkg
Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
using MLCourse,Plots, DataFrames, CSV,StatsPlots ,MLJ, MLJLinearModels, Random,LinearAlgebra,MLJMultivariateStatsInterface,Statistics
end

# ╔═╡ 74fc57f3-a9a8-4851-a770-2e7269313acb
begin
	#Data import
	training = DataFrame(CSV.read(joinpath(@__DIR__, "data", "trainingdata.csv"), DataFrame))
	test = CSV.read(joinpath(@__DIR__, "data", "testdata.csv"), DataFrame)
	
end

# ╔═╡ ddd736b5-7684-4c81-9a33-5eb4c6b34565
training

# ╔═╡ 34199593-ee02-43b9-94ae-d9cef0af83b1
md"Let's first see the training set without any \"modification\" "

# ╔═╡ bcc73944-a3bd-449c-b5cb-88c6a1150cc0
describe(training)

# ╔═╡ 629c1bf2-5d6c-4b07-b8a0-ed922053a469
md"Now we replace the missing data"

# ╔═╡ 013fa3cf-f367-41e4-b74e-cb3e82efbb9d
coerce!(training, :precipitation_nextday => Binary);

# ╔═╡ e4ab0a1c-8c81-4a2d-94b7-367e94a00fa4
train_wm = MLJ.transform(fit!(machine(FillImputer(), training),verbosity = 2), training)

# ╔═╡ a3243444-47cd-4f28-97dd-d7e6c4642508
dropmissing!(train_wm);

# ╔═╡ ec5a2075-97f4-40d1-ba37-c1bd5c54bec5
#standardizer
begin
	mach_stand = machine(Standardizer(),train_wm)
	fit!(mach_stand,verbosity = 2)
end

# ╔═╡ 71d31380-bd3b-46b4-8374-e7e42fc0a12f
md"By running this we get that ALT\_sunshine\_4 mean have always value 0 "

# ╔═╡ 52b0f852-4b18-4cb2-9277-105b43a85aae
histogram( train_wm.precipitation_nextday,nbins = 2,normalize=true)

#Plot the output to see the propotion


# ╔═╡ 494301ee-0cb8-46c1-9139-62b07fab1803
md"Note for the following plots: Green = true and Orange = false"

# ╔═╡ c7bcd73a-acac-4ad4-b02a-94876b41547e
scatter( train_wm.GVE_wind_direction_1,train_wm.SMA_delta_pressure_4, c=int.(train_wm.precipitation_nextday).+1)

# ╔═╡ 68903f77-2546-4e22-a61f-6c4533b28baf
scatter( train_wm.PUY_sunshine_1,train_wm.PUY_sunshine_4, c=int.(train_wm.precipitation_nextday).+1)

# ╔═╡ 3cdad300-14de-4d17-9350-58fe90043f10
md"Correlation matrix"

# ╔═╡ 3b489503-3b88-4696-a787-15ce69ecf856
cor( Matrix(train_wm))

# ╔═╡ e46c74ba-6a11-430d-8347-b25798444353
md" ## Preparation of the test and train dataset for the other notebook/scripts"

# ╔═╡ d54de815-d624-4215-9023-ee8543aa7dd4
function data_split(data;
                        shuffle = false,
                        idx_train = 1:50,
                        idx_valid = 51:100,
                        idx_test = 101:500)
        idxs = if shuffle
                randperm(MersenneTwister(1234),size(data, 1))
            else
                1:size(data, 1)
            end
        (train = data[idxs[idx_train], :],
         valid = data[idxs[idx_valid], :],
         test = data[idxs[idx_test], :])
    end

# ╔═╡ 6f788532-fa05-4d7f-bbfd-bf8e5ae4dc8a
begin
split_d1=data_split(train_wm;shuffle = true,idx_train = 1:trunc(Int,nrow(train_wm)*0.8),idx_valid=1:1,idx_test =(trunc(Int,nrow(train_wm)*0.8)+1):nrow(train_wm));
	#Without validation set
	train_data_nv = split_d1.train;
	test_data_nv = split_d1.test;
end	;

# ╔═╡ a9580e50-ecd0-484a-b163-1c2c272b2fc3
begin
#save the train and test_files 
CSV.write(joinpath(@__DIR__, "data", "train_data_nv.csv"), train_data_nv)
CSV.write(joinpath(@__DIR__, "data", "test_data_nv.csv"), test_data_nv)
end

# ╔═╡ 473ea8e4-7977-4107-a624-e2aff21fbae4
md"## PCA"

# ╔═╡ aaa040fc-bfcc-4b03-a803-35a5ce2ad9c1
md"With normalization"

# ╔═╡ 54156f6b-1225-4188-8ec3-9f7273532ef9
pca_data = fit!(machine(@pipeline(Standardizer(features = [:ALT_sunshine_4], ignore = true), PCA()), 
		select(train_wm,Not(:precipitation_nextday))),verbosity = 2);

# ╔═╡ 51f2a033-d7d5-46fb-864a-e13fe0df4e03
biplot(pca_data)

# ╔═╡ c00c9286-cea5-4c50-b8f3-4a2c2ecc5211
report(pca_data)

# ╔═╡ Cell order:
# ╠═f2bfbb40-5f40-11ec-18f2-d12eb2303d46
# ╠═74fc57f3-a9a8-4851-a770-2e7269313acb
# ╠═ddd736b5-7684-4c81-9a33-5eb4c6b34565
# ╟─34199593-ee02-43b9-94ae-d9cef0af83b1
# ╠═bcc73944-a3bd-449c-b5cb-88c6a1150cc0
# ╟─629c1bf2-5d6c-4b07-b8a0-ed922053a469
# ╠═013fa3cf-f367-41e4-b74e-cb3e82efbb9d
# ╠═e4ab0a1c-8c81-4a2d-94b7-367e94a00fa4
# ╠═a3243444-47cd-4f28-97dd-d7e6c4642508
# ╠═ec5a2075-97f4-40d1-ba37-c1bd5c54bec5
# ╟─71d31380-bd3b-46b4-8374-e7e42fc0a12f
# ╠═52b0f852-4b18-4cb2-9277-105b43a85aae
# ╟─494301ee-0cb8-46c1-9139-62b07fab1803
# ╠═c7bcd73a-acac-4ad4-b02a-94876b41547e
# ╠═68903f77-2546-4e22-a61f-6c4533b28baf
# ╟─3cdad300-14de-4d17-9350-58fe90043f10
# ╠═3b489503-3b88-4696-a787-15ce69ecf856
# ╟─e46c74ba-6a11-430d-8347-b25798444353
# ╠═d54de815-d624-4215-9023-ee8543aa7dd4
# ╠═6f788532-fa05-4d7f-bbfd-bf8e5ae4dc8a
# ╠═a9580e50-ecd0-484a-b163-1c2c272b2fc3
# ╟─473ea8e4-7977-4107-a624-e2aff21fbae4
# ╟─aaa040fc-bfcc-4b03-a803-35a5ce2ad9c1
# ╠═54156f6b-1225-4188-8ec3-9f7273532ef9
# ╠═51f2a033-d7d5-46fb-864a-e13fe0df4e03
# ╠═c00c9286-cea5-4c50-b8f3-4a2c2ecc5211
