using Distributions
using LinearAlgebra
using CovarianceEstimation

function hill_weights(w, K, h)
    wts= (w.^h ./ (K.^h .+ w.^h))
    return StatsBase.weights(wts)
end

function init(models, np, rho)
    d = Inf
    count = 1
    m = sample(range(1, stop=length(models)))
    pars = rand.(models[m])
    d = rho[m](pars)
    return vcat(m, pars, fill(0, maximum(np) - np[m]), d, count)
end

# SMC sampler (for subsequent iterations)
function cont(models, pts, wts, np, i, ker, rho)
    d = Inf
    count = 1

  # while d==Inf
    m = sample(1:length(models))
    while size(pts[m,i - 1])[2] == 0
        m = sample(1:length(models))
    end

    params = pts[m,i - 1][:,sample(1:size(pts[m,i - 1])[2], weights(wts[m,i - 1]))]
    params = params + rand(ker[m])
    while prod(pdf.(models[m], params)) == 0
        m = sample(1:length(models))
        while size(pts[m,i - 1])[2] == 0
            m = sample(1:length(models))
        end
        count = count + 1
        params = pts[m,i - 1][:,sample(1:size(pts[m,i - 1])[2], weights(wts[m,i - 1]))]
        params = params + rand(ker[m])
    end
    d = rho[m](params)
  # end
    return vcat(m, params, fill(0, maximum(np) - np[m]), d, count)
end



function APMC_lite(N, models, rho,;names=Vector[[string("parameter", i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.02,n=2)
    i = 1
    i2 = 1
    lm = length(models)
    s = round(Int, N * prop)
  # array for number of parameters in each model
    np = Array{Int64}(undef, length(models))
    for j in 1:lm
        np[j] = length(models[j])
    end
  # array for SMC kernel used in weights
    ker = Array{Any}(undef, lm)
    template = Array{Any}(undef, lm, 1)
  # particles array
    pts = similar(template)
  # covariance matrix array
    sig = similar(template)
  # weights array
    wts = similar(template)
  # model probability at each iteration array
    p = zeros(lm, 1)

    temp = zeros(maximum(length.(models))+3,N)
    @threads for j in 1:N
        temp[:,j] .= init(models, np, rho)
    end

    its = [sum(temp[size(temp)[1],:])]
    epsilon = [quantile(collect(temp[maximum(np) + 2,:]), prop)]
    pacc = ones(lm, 1)
    println(round.([epsilon[i];its[i]], digits=3))
    temp = temp[:,temp[maximum(np) + 2,:] .<= epsilon[i]]
    temp = temp[:,1:s]
    for j in 1:lm
        pts[j,i2] = temp[2:(np[j] + 1),temp[1,:] .== j]
        wts[j,i2] = StatsBase.weights(fill(1.0, sum(temp[1,:] .== j)))
    end
    
    dists = transpose(temp[(maximum(np) + 2),:])
    for j in 1:lm
        p[j] = sum(wts[j,1])
    end
    for j in 1:lm
        # sig[j,i2] = cov(pts[j,i2], wts[j,i2], 2, corrected = false)
        params = pts[j,i2][:,sample(1:size(pts[j,i2])[2], wts[j,i2], N)]'
        sig[j,i2] = CovarianceEstimation.cov(CovarianceEstimation.LinearShrinkage(target=CovarianceEstimation.DiagonalUnequalVariance(), shrinkage=:lw), params)
    end
    p = p ./ sum(p)
    nbs = Array{Integer}(undef, length(models))
    for j in 1:lm
        nbs[j] = length(wts[j,i2])
        println(round.(hcat(mean(diag(sig[j,i2])[1:(np[j])]), pacc[j,i2], nbs[j], p[j,i2]), digits=3))
    end
    while maximum(pacc[:,i]) > paccmin
        if i <= 2
            pts = reshape(pts, i * length(models))
            sig = reshape(sig, i * length(models))
            wts = reshape(wts, i * length(models))
            for j in 1:length(models)
                push!(pts, Array{Any}(undef, 1))
                push!(sig, Array{Any}(undef, 1))
                push!(wts, Array{Any}(undef, 1))
            end
            pts = reshape(pts, length(models), i + 1)
            sig = reshape(sig, length(models), i + 1)
            wts = reshape(wts, length(models), i + 1)
        end
        i = i + 1
        i2 = i2 + 1

        if i2 > 3
            i2 = 3
            pts = view(pts, :, [1,3,2])
            wts = view(wts, :, [1,3,2])
            sig = view(sig, :, [1,3,2])
        end

        for j in 1:lm
            ker[j] = MvTDist(1, fill(0.0, np[j]), float.(n * sig[j,i2 - 1]))
        end

        temp2 = zeros(maximum(length.(models))+3,N-s)
        @threads for j in (1:(N - s))
            temp2[:,j] .= cont(models, pts, wts, np, i2, ker, rho)
        end

        its = vcat(its, sum(temp2[size(temp2)[1],:]))
        temp = hcat(temp, temp2)
        inds = sortperm(reshape(temp[maximum(np) + 2,:], N))[1:s]
        temp = temp[:,inds]
        dists = transpose(temp[(maximum(np) + 2),:])
        epsilon = vcat(epsilon, temp[(maximum(np) + 2),s])
        pacc = hcat(pacc, zeros(lm))
        for j in 1:lm
            if sum(temp2[1,:] .== j) > 0
                pacc[j,i] = sum(temp[1,inds .> s] .== j)  / sum(temp2[1,:] .== j)
            else pacc[j,i] == 0
            end
            # if nbs[j] == N/2
            #     samp = ABCfit(pts, sig, wts, p, its, dists, epsilon, temp, pacc, names, models)
            #     return(samp)
            # end
        end
        println(round.(vcat(epsilon[i], its[i]), digits=3))
        for j in 1:lm
            pts[j,i2] = temp[2:(np[j] + 1),temp[1,:] .== j]
            if size(pts[j,i2])[2] > 0
                keep = inds[reshape(temp[1,:] .== j, s)] .<= s
                #wts[j,i2 - 1] = hill_weights(convert(Vector, wts[j,i2 - 1]), quantile(convert(Vector, wts[j,i2 - 1]), 0.9), 2.0)

                wts[j,i2] = vcat([if !keep[k]
                    prod(pdf.(models[j], (pts[j,i2][:,k]))) / (1 / (sum(wts[j,i2 - 1])) * dot(convert(Vector, wts[j,i2 - 1]), pdf(ker[j], broadcast(-, pts[j,i2 - 1], pts[j,i2][:,k]))))
                else
                    0.0
                end for k in 1:length(keep)])
                
                if length(wts[j,i2]) == 1
                    wts[j,i2] = fill(wts[j,i2], 1)
                end
                l = 1
                for k in 1:length(keep)
                    if keep[k]
                        wts[j,i2][k] = wts[j,i2 - 1][l]
                        l = l + 1
                    end
                end
                if length(wts[j,i2]) > 1
                    wts[j,i2] = StatsBase.weights(wts[j,i2])
                end
            else
                wts[j,i2] = zeros(0)
            end
            
        end
        p = hcat(p, zeros(length(models)))
        for j in 1:lm
            p[j,i] = sum(wts[j,i2])
        end
        for j in 1:lm
            if (size(pts[j,i2])[2] > np[j])
                params = pts[j,i2][:,sample(1:size(pts[j,i2])[2], wts[j,i2], N)]'
                # params = zeros(N, np[j])
                # wts_t = hill_weights(convert(Vector, wts[j,i2]), quantile(convert(Vector, wts[j,i2]), 0.9), 2.0)
                # for num = 1:N
                #     params[num,:] = pts[j,i2][:,sample(1:size(pts[j,i2])[2], wts_t)]
                # end
                sig[j,i2] = CovarianceEstimation.cov(CovarianceEstimation.LinearShrinkage(target=CovarianceEstimation.DiagonalUnequalVariance(), shrinkage=:lw), params)
  
                if isposdef(n * sig[j,i2])
                    dker = MvTDist(1, pts[j,i2 - 1][:,1], float.(n * sig[j,i2]))
                    if pdf(dker, pts[j,i2][:,1]) == Inf
                        sig[j,i2] = sig[j,i2 - 1]
                    end
                else
                    sig[j,i2] = sig[j,i2 - 1]
                end
            else
                sig[j,i2] = sig[j,i2 - 1]
            end
        end
        p[:,i] = p[:,i] ./ sum(p[:,i])
        for j in 1:lm
            nbs[j] = length(wts[j,i2])
            println(round.(hcat(mean(diag(sig[j,i2]) ./ diag(sig[j,1])), pacc[j,i], nbs[j], p[j,i]), digits=5))
        end

        if (i > 11 && abs(epsilon[i] - epsilon[i-10]) < 1e-3) || i > 2000
            samp = ABCfit(pts, sig, wts, p, its, dists, epsilon, temp, pacc, names, models)
            return(samp)
        end

    end
    samp = ABCfit(pts, sig, wts, p, its, dists, epsilon, temp, pacc, names, models)
    return(samp)
end


function APMC(N, models, rho,;names=Vector[[string("parameter", i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.01,n=2,covar="LinearShrinkage(DiagonalUnequalVariance(), :lw)",perturb="Cauchy")
    i = 1
    scoot=0
    lm = length(models)
    s = round(Int, N * prop)
  # array for number of parameters in each model
    np = Array{Int64}(undef, length(models))
    for j in 1:lm
        np[j] = length(models[j])
    end
  # array for SMC kernel used in weights
    ker = Array{Any}(undef, lm)
    template = Array{Any}(undef, lm, 1)
  # particles array
    pts = similar(template)
  # covariance matrix array
    sig = similar(template)
  # weights array
    wts = similar(template)
  # model probability at each iteration array
    p = zeros(lm, 1)
    temp = zeros(maximum(length.(models))+3,N)
    @threads for j in 1:N
        temp[:,j] .= init(models, np, rho)
    end
    its = [sum(temp[size(temp)[1],:])]
    epsilon = [quantile(collect(temp[maximum(np) + 2,:]), prop)]
    pacc = ones(lm, 1)
    println(round.([epsilon[i];its[i]], digits=3))
    temp = temp[:,temp[maximum(np) + 2,:] .<= epsilon[i]]
    temp = temp[:,1:s]
    for j in 1:lm
        pts[j,i] = temp[2:(np[j] + 1),temp[1,:] .== j]
        wts[j,i] = StatsBase.weights(fill(1.0, sum(temp[1,:] .== j)))
    end
    dists = transpose(temp[(maximum(np) + 2),:])
    for j in 1:lm
        p[j] = sum(wts[j,1])
    end
    for j in 1:lm
        params = zeros(N, np[j])
        for num = 1:N
            params[num,:] = pts[j,i][:,sample(1:size(pts[j,i])[2], wts[j,i])]
        end
        sig[j,i] = CovarianceEstimation.cov(eval(Meta.parse("$covar")), params)
    end
    p = p ./ sum(p)
    nbs = Array{Integer}(undef, length(models))
    for j in 1:lm
        nbs[j] = length(wts[j,i])
        println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]), pacc[j,i], nbs[j], p[j,i]), digits=3))
    end
    while maximum(pacc[:,i]) > paccmin
        pts = reshape(pts, i * length(models))
        sig = reshape(sig, i * length(models))
        wts = reshape(wts, i * length(models))
        for j in 1:length(models)
            push!(pts, Array{Any}(undef, 1))
            push!(sig, Array{Any}(undef, 1))
            push!(wts, Array{Any}(undef, 1))
        end
        pts = reshape(pts, length(models), i + 1)
        sig = reshape(sig, length(models), i + 1)
        wts = reshape(wts, length(models), i + 1)
        i = i + 1
        for j in 1:lm
            if perturb == "Cauchy"
                ker[j] = MvTDist(1, fill(0.0, np[j]), float.(n * sig[j,i-1]))
            elseif perturb == "Normal"
                ker[j] = MvNormal(fill(0.0, np[j]), n * sig[j,i - 1])
            end
        end
        temp2 = zeros(maximum(length.(models))+3,N-s)
        @threads for j in (1:(N - s))
            temp2[:,j] .= cont(models, pts, wts, np, i, ker, rho)
        end
        its = vcat(its, sum(temp2[size(temp2)[1],:]))
        temp = hcat(temp, temp2)

        # if i > 5 && scoot == 0
        #     scoot = 1
        #     temp[2:48,1] = readdlm("midpoint_parameters_59.txt")[:,1]
        #     temp[49,1] = rho[1](temp[2:48,1])
        #     temp[2:48,2] = readdlm("midpoint_parameters_15.txt")[:,1]
        #     temp[49,2] = rho[1](temp[2:48,2])
        #     temp[2:48,3] = readdlm("midpoint_parameters_48.txt")[:,1]
        #     temp[49,3] = rho[1](temp[2:48,3])
        #     temp[2:48,4] = readdlm("midpoint_parameters_76.txt")[:,1]
        #     temp[49,4] = rho[1](temp[2:48,4])
        #     temp[2:48,5] = readdlm("midpoint_parameters_77.txt")[:,1]
        #     temp[49,5] = rho[1](temp[2:48,5])
        #     temp[2:48,6] = readdlm("midpoint_parameters_78.txt")[:,1]
        #     temp[49,6] = rho[1](temp[2:48,6])
        #     temp[2:48,7] .= readdlm("midpoint_parameters_9.txt")[:,1]
        #     temp[49,7] = rho[1](temp[2:48,7])
        # end

        inds = sortperm(reshape(temp[maximum(np) + 2,:], N))[1:s]
        temp = temp[:,inds]
        dists = hcat(dists, transpose(temp[(maximum(np) + 2),:]))
        epsilon = vcat(epsilon, temp[(maximum(np) + 2),s])
        pacc = hcat(pacc, zeros(lm))
        for j in 1:lm
            if sum(temp2[1,:] .== j) > 0
                pacc[j,i] = sum(temp[1,inds .> s] .== j) / sum(temp2[1,:] .== j)
            else pacc[j,i] == 0
            end
        end
        println(round.(vcat(epsilon[i], its[i]), digits=3))
        for j in 1:lm
            pts[j,i] = temp[2:(np[j] + 1),temp[1,:] .== j]
            if size(pts[j,i])[2] > 0
                keep = inds[reshape(temp[1,:] .== j, s)] .<= s
                wts[j,i] = vcat([if !keep[k]
                    prod(pdf.(models[j], (pts[j,i][:,k]))) / (1 / (sum(wts[j,i - 1])) * dot(convert(Vector, wts[j,i - 1]), pdf(ker[j], broadcast(-, pts[j,i - 1], pts[j,i][:,k]))))
                else
                    0.0
                end for k in 1:length(keep)])
                
                if length(wts[j,i]) == 1
                    wts[j,i] = fill(wts[j,i][1], 1)
                end
                l = 1
                for k in 1:length(keep)
                    if keep[k]
                        wts[j,i][k] = wts[j,i - 1][l]
                        l = l + 1
                    end
                end
                if length(wts[j,i]) > 1
                    wts[j,i] = StatsBase.weights(wts[j,i])
                end
            else
                wts[j,i] = zeros(0)
            end
        end

        p = hcat(p, zeros(length(models)))
        for j in 1:lm
            if length(wts[j,i]) == 1
                wts[j,i] = weights(wts[j,i])
                p[j,i] = sum(wts[j,i][1])
            else
                wts[j,i] = weights(wts[j,i])
                p[j,i] = sum(wts[j,i])
            end
        end
        for j in 1:lm
            if(size(pts[j,i])[2] > np[j])
                params = pts[j,i][:,sample(1:size(pts[j,i])[2], wts[j,i], N)]'
                sig[j,i] = CovarianceEstimation.cov(eval(Meta.parse("$covar")), params)
                @show isposdef(sig[j,i])
                if isposdef(sig[j,i])
                    if perturb == "Cauchy"
                        dker = MvTDist(1, pts[j,i - 1][:,1], float.(n * sig[j,i-1]))
                    elseif perturb == "Normal"
                        dker = MvNormal(pts[j,i - 1][:,1], n * sig[j,i - 1])
                    end
                    if pdf(dker, pts[j,i][:,1]) == Inf
                        sig[j,i] = sig[j,i - 1]
                    end
                else
                    sig[j,i] = sig[j,i - 1]
                end
            else
                sig[j,i] = sig[j,i - 1]
            end
            # if i > 10
            #     display(plot(histogram(temp[49,:]),bar(diag(sig[j,i]))))
            # end
        end
        p[:,i] = p[:,i] ./ sum(p[:,i])
        for j in 1:lm
            nbs[j] = length(wts[j,i])
            println(round.(hcat(mean(diag(sig[j,i]) ./ diag(sig[j,1])), pacc[j,i], nbs[j], p[j,i]), digits=3))
        end
        if (i > 11 && abs(epsilon[i] - epsilon[i-5]) < 1e-3)
            samp = ABCfit(pts, sig, wts, p, its, dists, epsilon, temp, pacc, names, models)
            return(samp)
        end
    end
    samp = ABCfit(pts, sig, wts, p, its, dists, epsilon, temp, pacc, names, models)
    return(samp)
end

