using MKL
using Base.Threads
using StatsBase, Random
using Distributions
using DelimitedFiles
using LinearAlgebra
using OrdinaryDiffEq
using CSV, DataFrames
using Interpolations
using Plots

atol=1e-6
rtol=1e-6
#data = CSV.read("HXTs_for_fitting.csv", DataFrame)
data = CSV.read("mean_HXTs_for_fitting.csv",DataFrame)

function wt(params)
    params
end

function mth1ko(params)
    pars = copy(params)
    pars[12] = -(pars[17]+pars[26])
    return collect(pars)
end

function std1ko(params)
    pars = copy(params)
    pars[18] = 0.0
    return collect(pars)
end	

function mig2ko(params)
    pars = copy(params)
    pars[46] = 0.0
    return collect(pars)
end	

function mig1ko(params)
    pars = copy(params)
    pars[22] = 0.0 #imig1
    return collect(pars)
end	

function rgt2ko(params)
    pars = copy(params)
    pars[11] = 0.0 #dmth1rgt2
    pars[21] = 0.0 #thrgt2#added by Luis 20201109
    return collect(pars)
end

function snf3ko(params)
    pars = copy(params)
    pars[10] = - pars[43] #0.0 #thsnf3
    return collect(pars)
end

# Intracellular glucose assuming logistic growth
function intracellular_glucose(t, g, r)
    gmax = maximum(g)
    updown_indices = findall(x -> x > 1e-6 && x < gmax - 1e-6, g)

    if length(updown_indices) < 2
        # Handle step-like time series
        up_i = findfirst(x -> x > 1e-6, g)
        down_i = length(t)
    else
        # Handle hat-like time series
        sep_i = findmax(diff(updown_indices))[2]
        up_t = mean(t[updown_indices[1:sep_i]])
        up_i = argmin((t .- up_t) .^ 2.0)
        down_t = mean(t[updown_indices[sep_i+1:end]])
        down_i = argmin((t .- down_t) .^ 2.0)
    end
    gmin = 1e-7
    gi =  gmin .* ones(length(t))
    gi_in_glucose = gmax ./ (1 .+ (gmax .- gmin) ./ gmin .* exp.(-r .* (t .- t[up_i])))
    gi[up_i:down_i] = gi_in_glucose[up_i:down_i]
    gi_fun = LinearInterpolation(t, gi)
    return gi_fun
end


function ss1(h0,p)
    imig1 = p[22] 
    if p[18] > 0
        std1tot = p[26] + p[17]
    else 
        std1tot = 0.0
    end
    if imig1 == 0.0
        function f1(du,u,p,t)
            dmth1 = p[7]; # [8]
            thmth1 = p[12]; # [13]
            kmth1mig2 = p[15];# [16]
            nmth1mig2 = p[16]; # [17]
            dmig2 = p[25]; # [26]
            kmig2std1 = p[26];# [28]
            nmig2std1 = p[27];# [29]
            kmig2mth1 = p[28]; # [30]
            nmig2mth1 = p[29]; # [31]
            thmig2 = p[45];
            smig2 = p[46];

            Mig2 = u[1] 
            Mth1 = u[2] 

            smth1= dmth1*(std1tot + thmth1)
            dmig2g = dmig2 * 10^thmig2 
            
            Std1 = std1tot
            du[1] = +(smig2 / (1 + (Mth1 / kmig2mth1)^nmig2mth1 + (Std1 / kmig2std1)^nmig2std1)) - dmig2 * Mig2 - dmig2g * Mig2
            du[2] = +(smth1 / (1 + (Mig2 / kmth1mig2)^nmth1mig2)) - dmth1 * Mth1
        end
        u0 = zeros(2)
        prob = ODEProblem(f1,u0,(0.0,1e6),p)
        sol = solve(prob, TRBDF2(), save_everystep = false, verbose = false, abstol = atol, reltol = rtol,dtmin=1e-12)
        Mig10 = 0.0
        Mig20 = sol[1,end] 
        Mth10 = max(sol[2,end],0)
    else
        function f2(du,u,p,t)
            ksnf1std1 = p[4];# [4]
            nsnf1 = p[5];# [5]
            dmth1 = p[7];# [8]
            thmth1 = p[12];# [13]
            kmth1mig1 = p[13];# [14]
            nmth1mig1 = p[14];# [15]
            kmth1mig2 = p[15];# [16]
            nmth1mig2 = p[16];# [17]
            imig1 = p[22];# [23]
            kmig1snf1 = p[23];# [24]
            theta2 = p[24];# [25]
            dmig2 = p[25];# [26]
            kmig2std1 = p[26];# [30]
            nmig2std1 = p[27];# [31]
            kmig2mth1 = p[28];# [32]
            nmig2mth1 = p[29];# [33]
            ell= p[44];# [48]
            thmig2 = p[45];
            smig2 = p[46];

            Mig1 = u[1]
            Mig2 = u[2] 
            Mth1 = u[3] 

            # parameter thmth1 replaces smth1 - imposes Mth1 is usually greater than Std1 
            smth1= dmth1*(std1tot + thmth1)
            emig1max = (10^theta2)*imig1
            dmig2g = dmig2 * 10^thmig2 

            Std1 = std1tot
            Snf1 = (1 + Std1 / ksnf1std1)^nsnf1 / ((1 + Std1 / ksnf1std1)^nsnf1 + ell)
            du[1] = +(imig1 * (1 - Mig1)) - (emig1max * Snf1 * Mig1 / (kmig1snf1 + Mig1))
            du[2] = +(smig2 / (1 + (Mth1 / kmig2mth1)^nmig2mth1 + (Std1 / kmig2std1)^nmig2std1)) - dmig2 * Mig2 - dmig2g * Mig2
            du[3] = +(smth1 / (1 + (Mig1 / kmth1mig1)^nmth1mig1 + (Mig2 / kmth1mig2)^nmth1mig2)) - dmth1 * Mth1
        end
        u0 = zeros(3)
        prob = ODEProblem(f2,u0,(0.0,1e6),p)
        sol = solve(prob, TRBDF2(), save_everystep = false, verbose = false, abstol = atol, reltol = rtol,dtmin=1e-12)
        Mig10 = sol[1,end] 
        Mig20 = sol[2,end] 
        Mth10 = max(sol[3,end],0)
    end
    if p[18] > 0
        std1tot = p[26] + p[17]
    else 
        std1tot = 0.0
    end
    return [h0;Mig10;Mig20;Mth10;std1tot;0.0;0.0]
end


function makeproblem1(intraceullar_glucose,lg, t, h0, genotype)
    # making a general spline for every purpose
        function model(dydt, y, parameters, t)
            g = lg(t)
            ig = intraceullar_glucose(t)
            j = 1
            k3 = parameters[j]; j += 1;# [1]
            thk = parameters[j]; j += 1;# [2]
            ksnf1 = parameters[j]; j += 1;# [3]
            ksnf1std1 = parameters[j]; j += 1;# [4]
            nsnf1 = parameters[j]; j += 1;# [5]
            nsnf2 = parameters[j]; j += 1;# [6]
            dmth1 = parameters[j]; j += 1;# [7]
            nmth1snf3 = parameters[j]; j += 1;# [8]
            nmth1rgt2 = parameters[j]; j += 1;# [9]
            thsnf3 = parameters[j]; j += 1;# [10]
            dmth1rgt2 = parameters[j]; j += 1;# [11]
            thmth1 = parameters[j]; j += 1;# [12]
            kmth1mig1 = parameters[j]; j += 1;# [13]
            nmth1mig1 = parameters[j]; j += 1;# [14]
            kmth1mig2 = parameters[j]; j += 1;# [15]
            nmth1mig2 = parameters[j]; j += 1;# [16]
            thstd1 = parameters[j]; j += 1;# [17]
            istd1 = parameters[j]; j += 1;# [18]
            nstd1 = parameters[j]; j += 1;# [19]
            nstd3 = parameters[j]; j += 1;# [20]
            thrgt2 = parameters[j]; j += 1;# [21]
            imig1 = parameters[j]; j += 1;# [22]
            kmig1snf1 = parameters[j]; j += 1;# [23]
            theta2 = parameters[j]; j += 1;# [24]
            dmig2 = parameters[j]; j += 1;# [25]
            kmig2std1 = parameters[j]; j += 1;# [26]
            nmig2std1 = parameters[j]; j += 1;# [27]
            kmig2mth1 = parameters[j]; j += 1;# [28]
            nmig2mth1 = parameters[j]; j += 1;# [29]
            dhxt4 = parameters[j]; j += 1;# [30]
            dhxt4max = parameters[j]; j += 1;# [31]
            kdhxt4 = parameters[j]; j += 1;# [32]
            ndhxt4 = parameters[j]; j += 1;# [33]
            shxt4 = parameters[j]; j += 1;# [34]
            th = parameters[j]; j += 1;# [35]
            nhxt4mth1 = parameters[j]; j += 1;# [36]
            khxt4std1 = parameters[j]; j += 1;# [37]
            nhxt4std1 = parameters[j]; j += 1;# [38]
            khxt4mig1 = parameters[j]; j += 1;# [39]
            khxt4mig2 = parameters[j]; j += 1;# [40]
            nhxt4mig1 = parameters[j]; j += 1;# [41]
            nhxt4mig2 = parameters[j]; j += 1;# [42]
            estd1max3 = parameters[j]; j += 1;# [43]
            ell= parameters[j];j += 1;# [44]
            thmig2 = parameters[j];j += 1;# [45]
            smig2 = parameters[j];j += 1;# [46]
            kdmig2 = parameters[j];j += 1;# [47]
            mdmig2 = parameters[j];#48

            Hxt4 = y[1]
            Mig1 = y[2] 
            Mig2 = y[3] 
            Mth1 = y[4] 
            Std1 = y[5]
            y[6] = g
            y[7] = ig

            if istd1 == 0.0
                std1tot = 0.0
            else
                std1tot = kmig2std1 + thstd1
            end

            smth1= dmth1*(std1tot + thmth1)

            k2= k3 + thk
            estd1max2= dmth1rgt2+ thrgt2
            dmth1snf3= estd1max3 + thsnf3
            emig1max = (10^theta2)*imig1
            dmig2g = dmig2 * 10^thmig2 

            khxt4mth1 = (smth1/dmth1)*exp(-th)

            if k2 > 10^4 || k2 < 10^(-4) 
                error()
            end

            if dmth1snf3 !== 0.0
                if dmth1snf3 < 10^(-4)|| dmth1snf3 > 10^2
                    error()
                end
            end

            if estd1max2 !== 0.0
                if estd1max2 < 10^(-2) || estd1max2 > 10^2 
                    error()
                end
            end
            
            if emig1max !== 0.0
                if emig1max < 10^(-2) || emig1max > 10^2
                    error()
                end
            end
            
            if smth1 !== 0.0
                if smth1 < 10^(-2) || smth1 > 10^2
                    error()
                end
            end

            if std1tot !== 0.0
                if std1tot > 1 
                    error()
                end
            end

            if smth1/dmth1 > 10.0
                error()
            end


            Snf1 = (1 + Std1 / ksnf1std1)^nsnf1 / ((1 + Std1 / ksnf1std1)^nsnf1 + ell*(1 + ig / ksnf1)^nsnf2)
            if smth1 == 0 
                dydt[1] = +(shxt4 / (1 + (Std1 / khxt4std1)^nhxt4std1 + (Mig1 / khxt4mig1)^nhxt4mig1 + (Mig2 / khxt4mig2)^nhxt4mig2)) - dhxt4 * Hxt4 - (dhxt4max * Hxt4) / (1 + (ig / kdhxt4)^ndhxt4) 
            else
                dydt[1] = +(shxt4 / (1 + (Mth1 / khxt4mth1)^nhxt4mth1 + (Std1 / khxt4std1)^nhxt4std1 + (Mig1 / khxt4mig1)^nhxt4mig1 + (Mig2 / khxt4mig2)^nhxt4mig2)) - dhxt4 * Hxt4 - (dhxt4max * Hxt4) / (1 + (ig / kdhxt4)^ndhxt4) 
            end
            dydt[2] = +(imig1 * (1 - Mig1)) - (emig1max * Snf1 * Mig1 / (kmig1snf1 + Mig1))
            dydt[3] = +(smig2 / (1 + (Mth1 / kmig2mth1)^nmig2mth1 + (Std1 / kmig2std1)^nmig2std1)) - dmig2 * Mig2 - dmig2g * Mig2 / (1 + (ig/kdmig2)^mdmig2)
            dydt[4] = +(smth1 / (1 + (Mig1 / kmth1mig1)^nmth1mig1 + (Mig2 / kmth1mig2)^nmth1mig2)) - dmth1 * Mth1 - (dmth1snf3 * g^nmth1snf3 / (k3^nmth1snf3 + g^nmth1snf3) * Mth1) - (dmth1rgt2 * g^nmth1rgt2 / (k2^nmth1rgt2 + g^nmth1rgt2) * Mth1)
            dydt[5] = +(istd1 * (std1tot - Std1)) - estd1max2 * g^nstd1 / (k2^nstd1 + g^nstd1) * Std1 - estd1max3 * g^nstd3 / (k3^nstd3 + g^nstd3) * Std1
            dydt[6] = 0.0
            dydt[7] = 0.0
        end

    prob1(params) = ODEProblem(model, ss1(h0,[j for j in genotype(params)]), t, [j for j in genotype(params)])
    
end

problems =[]
fluorescence = []
fluorescence_sem = []
tpoints = []
names=[]
glucose_interp = []
int_glucose_interp = []
for h in unique(data[!,:strain])[1:6]
    for g in unique(data[!,:experiment_type])
        temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
        temp_data2 = data[coalesce.(data.signal_type .== "fluorescence_sem" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
        temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
        if !isempty(temp_data1)
            temp_data4 = temp_data1[!, [1,3]]
            temp_data5 = temp_data2[!, [1,3]]
            temp_data6 = temp_data3[!, [1,3]]
            times = temp_data4[!,:time]
            fluor = temp_data4[!,:signal_value]
            fluor_sem = temp_data5[!,:signal_value]
            if minimum(times) !== 0.0
                glucose = LinearInterpolation(vcat(0.0,times),vcat(0.0,temp_data6[1:length(times),:signal_value]),extrapolation_bc=Line()) #some glucose time series miss out the first time point/value! this is my hack
                #int_glucose = intracellular_glucose(vcat(0.0,times),vcat(0.0,temp_data6[1:length(times),:signal_value]),50.0)
                int_glucose = intracellular_glucose(vcat(0.0,times),glucose(vcat(0.0,times)),50)
                push!(glucose_interp,glucose)
                push!(int_glucose_interp,int_glucose)
            else
                glucose = LinearInterpolation(times,temp_data6[1:length(times),:signal_value],extrapolation_bc=Line())
                #int_glucose = intracellular_glucose(times,temp_data6[1:length(times),:signal_value],50.0)
                int_glucose = intracellular_glucose(times,glucose(times),50)
                push!(glucose_interp,glucose)
                push!(int_glucose_interp,int_glucose)
            end

            if h =="HXT4_GFP"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))
                push!(fluorescence,fluor)
                push!(fluorescence_sem,fluor_sem)
                push!(tpoints,times)
                push!(names,h*"_$g")
            elseif h == "HXT4_GFP_snf3_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], snf3ko))
                push!(fluorescence,fluor)
                push!(fluorescence_sem,fluor_sem)
                push!(tpoints,times)
                push!(names,h*"_$g")
            elseif h == "HXT4_GFP_rgt2_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], rgt2ko))
                push!(fluorescence,fluor)
                push!(fluorescence_sem,fluor_sem)
                push!(tpoints,times)
                push!(names,h*"_$g")
            elseif h == "HXT4_GFP_std1_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], std1ko))
                push!(fluorescence,fluor)
                push!(fluorescence_sem,fluor_sem)
                push!(tpoints,times)
                push!(names,h*"_$g")
            elseif h == "HXT4_GFP_mig1_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mig1ko))
                push!(fluorescence,fluor)
                push!(fluorescence_sem,fluor_sem)
                push!(tpoints,times)
                push!(names,h*"_$g")
            elseif h == "HXT4_GFP_mth1_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mth1ko))
                push!(fluorescence,fluor)
                push!(fluorescence_sem,fluor_sem)
                push!(tpoints,times)
                push!(names,h*"_$g")
            end
        end
    end
end

unique_experiment = chop.(names,tail=20)
unique_experiment[4] = chop(unique_experiment[4],tail=1)
unique_experiment = unique(unique_experiment)
#using DifferentialEquations

function trysolve(prob, pars, datamean, datasem,times) # main function to solve a system with two solvers and return a large value if both fail.
    try
        sol = solve(prob(pars), TRBDF2(), saveat = times,tstops=times, verbose = false, abstol = atol, reltol = rtol, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])]
        lsqval = mean(((arr[1] - datamean).^2 ./ mean(datamean)))
        return lsqval, arr[4], arr[5]
    catch
        return 1e20, 1e20, 1e20, 1e20, 1e20
    end
end

function trysolvet(prob, pars, datamean, datasem,times) # main function to solve a system with two solvers and return a large value if both fail.
    try
        sol = solve(prob(pars), TRBDF2(), saveat = times,tstops=times, verbose = false, abstol = atol, reltol = rtol, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])]
        lsqval = arr
    catch
        lsqval = 1e20
    end
end

function trysolve2(prob, pars, datamean, datasem,times) # main function to solve a system with two solvers and return a large value if both fail.
    try
        sol = solve(prob(pars), TRBDF2(), saveat = times,tstops=times, verbose = true, abstol = 1e-12, reltol = 1e-12, dtmin = 1e-16,maxiters=1e9)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])]
        lsqval = mean(((arr[1] - datamean).^2 ./ mean(datamean)))
    catch
        lsqval = 1e20
    end
end


function rho_batch(d2)
    out = zeros(size(d2,2))
    @threads for jj in 1:size(d2,2)
        pars = copy(d2[:,jj])
        out[jj] = rho_lens(pars)
    end
    return out
end


custombounds101 =

[(-4.0,4.0), # k3 1
(-4.0,4.0), # k2 2
(-4.0, 4.0), # ksnf1 3 
(-4.0,4.0), # ksnf1std1 4
(log10(1.0), log10(5.0)), # nsnf1 5 
(log10(1.0), log10(5.0)), # nsnf2 6
(-4.0,2.0), # dmth1 7
(log10(1.0), log10(5.0)), # nmth1snf3 8
(log10(1.0), log10(5.0)), # nmth1rgt2 9
(-4.0,2.0), # dmth1snf3 10
(-4.0,2.0), # dmth1rgt2 11
(-2.0, 2.0), # thmth1 12
(-4.0,4.0), # kmig1mth1 13
(log10(1.0), log10(5.0)), # nmig1mth1 14
(-4.0,4.0), # kmig2mth1 15
(log10(1.0), log10(5.0)), # nmig2mth1 16
(-1.0,log10(100.0)), # thstd1 17
(-2.0,3.0), # istd1 18
(log10(1.0), log10(5.0)), # nstd1 19
(log10(1.0), log10(5.0)), # nstd3 20
(-2.0,3.0), # estd1max2 21
(-2.0,3.0), # imig1 22
(-4.0,4.0), # kmig1snf1 23
(-10.0, log10(4.0)), # theta2 24
(-4.0,2.0), # dmig2 25
(-4.0,4.0), #(log10(0.5),log10(3.0)),#al_std1 was originally (-3.0,4.0), # kmig2std1 26
(log10(1.0), log10(5.0)), # nmig2std1 27
(-4.0,4.0), # kmig2mth1 28
(log10(1.0), log10(5.0)), # nmig2mth1 29
(-4.0,2.0), # dhxt4 30, 1
(-4.0,2.0), # dhxt4max 31, 2
(-4.0,4.0), # kdhxt4 32, 3
(log10(1.0), log10(5.0)), # ndhxt4 33, 4
(-2.0,2.0), # shxt4 34, 5
(-10,log10(3)),#th (-4.0,4.0), # khxt4mth1 35, 6
(log10(1.0), log10(5.0)), # nhxt4mth1 36, 7
(-4.0,4.0),  #khxt4std1 37, 8
(log10(1.0), log10(5.0)), # nhxt4std1 38, 9
(-4.0,4.0), # khxt4mig1 39, 10
(-4.0,4.0), # khxt4mig2 40, 11
(log10(1.0), log10(5.0)),# nhxt4mig1 41, 12
(log10(1.0), log10(5.0)),# nhxt4mig2 42, 13
(-2.0, 3.0), # estd1max3 43
(-3.0, 3.0), # ell 44
(-1.0,log10(100.0)), #45
(-2.0, 2.0), # smig2 46
(-4.0, 4.0),# kdmig2 47
(log10(1.0),log10(5.0)),#mdmig2 48
(-1.0,2),#r1
(-1.0,2),#r2
(-1.0,2),#r3
(-1.0,2),#r4
(-1.0,2),#r5
(-1.0,2),#r6
(-1.0,2),#r7
(-1.0,2),#r8
(-1.0,2),#r9
(-1.0,2),#r10
(-1.0,2),#r11
(-1.0,2),#r12
(-1.0,2),#r13
(-1.0,2),#r14
(-1.0,2),#r15
(-1.0,2),#r16
(-1.0,2),#r17
(-1.0,2),#r18
(-1.0,2),#r19
(-1.0,2),#r20
(-1.0,2)] #r21

function rho_lens(d2)
    pars = copy(d2)
    pars = 10 .^ pars

    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])[1:6]
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                count = count + 1
                if minimum(times) !== 0.0
                    glucose = LinearInterpolation(vcat(0.0,times),vcat(0.0,temp_data6[1:length(times),:signal_value]),extrapolation_bc=Line()) #some glucose time series miss out the first time point/value! this is my hack
                    int_glucose = intracellular_glucose(vcat(0.0,times),glucose(vcat(0.0,times)),pars[48 + count])
                    push!(glucose_interp,glucose)
                    push!(int_glucose_interp,int_glucose)
                else
                    glucose = LinearInterpolation(times,temp_data6[1:length(times),:signal_value],extrapolation_bc=Line())
                    int_glucose = intracellular_glucose(times,glucose(times),pars[48 + count])
                    push!(glucose_interp,glucose)
                    push!(int_glucose_interp,int_glucose)
                end

                if h =="HXT4_GFP"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))
                elseif h == "HXT4_GFP_snf3_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], snf3ko))
                elseif h == "HXT4_GFP_rgt2_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], rgt2ko))
                elseif h == "HXT4_GFP_std1_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], std1ko))
                elseif h == "HXT4_GFP_mig1_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mig1ko))
                elseif h == "HXT4_GFP_mth1_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mth1ko))
                end
            end
        end
    end

    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 1
        E = 21.0 - length(errors) #mean(abs.(((out[3][2] ./ pars[36]) .^ pars[37] ./ ((out[3][3] ./ pars[38]) .^ pars[39])).- 1))
    else
        E = mean(errors) + 21.0 - length(errors) #mean(abs.(((out[3][2] ./ pars[36]) .^ pars[37] ./ ((out[3][3] ./ pars[38]) .^ pars[39])).- 1))
    end
    return E
end


using BlackBoxOptim
for i in 935:1000
    pars1 = readdlm("int_g_midpoint_parameters_846.txt")[:,1]
    pars1[35] = log10(1.175)
    @show rho_lens(pars1) 
    x0=pars1
    opt = BlackBoxOptim.bboptimize(rho_lens,x0;Method=:xnes,ini_sigma=0.1, SearchRange = custombounds101, MaxFuncEvals = 200000,NThreads=Threads.nthreads()-1)
    pars = best_candidate(opt)
    @show rho_lens(pars)
    writedlm("int_g_midpoint_parameters_$i.txt",pars)

    pars = 10 .^ pars

    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])[1:6]
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                count = count + 1
                if minimum(times) !== 0.0
                    glucose = LinearInterpolation(vcat(0.0,times),vcat(0.0,temp_data6[1:length(times),:signal_value]),extrapolation_bc=Line()) #some glucose time series miss out the first time point/value! this is my hack
                    int_glucose = intracellular_glucose(vcat(0.0,times),glucose(vcat(0.0,times)),pars[48 + count])
                    push!(glucose_interp,glucose)
                    push!(int_glucose_interp,int_glucose)
                else
                    glucose = LinearInterpolation(times,temp_data6[1:length(times),:signal_value],extrapolation_bc=Line())
                    int_glucose = intracellular_glucose(times,glucose(times),pars[48 + count])
                    push!(glucose_interp,glucose)
                    push!(int_glucose_interp,int_glucose)
                end

                if h =="HXT4_GFP"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))
                elseif h == "HXT4_GFP_snf3_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], snf3ko))
                elseif h == "HXT4_GFP_rgt2_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], rgt2ko))
                elseif h == "HXT4_GFP_std1_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], std1ko))
                elseif h == "HXT4_GFP_mig1_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mig1ko))
                elseif h == "HXT4_GFP_mth1_delta"
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mth1ko))
                end
            end
        end
    end


    using Plots
    out_s=[]
    out_m=[]
    out=[]
    #mkdir("output_$i")
    for jj = 1:length(problems)
        if names[jj] ∈ ["HXT4_GFP_step_glucose_0p01_percent","HXT4_GFP_step_glucose_0p1_percent","HXT4_GFP_step_glucose_1p0_percent"]
                 push!(out,ones(length(tpoints[jj])))
        else   
            sol = solve(problems[jj](pars[1:48]), TRBDF2(), saveat = tpoints[jj],tstops=tpoints[jj], verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
            arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])][1]
            push!(out,arr)
        end
    end

    plts = []
    count = 0
    gr(label="",grid=:on)
    ys =[(0,4.5),(0,3.5),(0,15),(0,10),(0,3.5),(0,3),(0,4.5)]
    for j in unique_experiment
        count+=1
        @show count
        if count ∈ [1,3,4,5,6,7]
            p1 = Plots.plot(tpoints[1+(count-1)*3:3+(count-1)*3],out[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",ylims=ys[count],linewidth=3.0)
            plot(p1,Plots.plot!(tpoints[1+(count-1)*3:3 .+ (count-1)*3],fluorescence[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",ylims=ys[count],linestyle=:dashdot))
            push!(plts,p1)
        end
    end
    display(plot(plts...,layout=(6,1),size=(500,1000)))
    savefig("HXT4_midpoint_$(i)_check.png")

    scatter(pars,label="$i",alpha=1.0)
    scatter!([10.0 ^ custombounds101[j][1] for j in 1:49],label="lower",alpha=0.5)
    scatter!([10.0 ^ custombounds101[j][2] for j in 1:49],label="upper",alpha=0.5,yscale=:log10,legend=:bottomleft)
    savefig("params_$(i)_bounds_check.png")

    out = trysolvet.(problems, [pars], fluorescence,fluorescence_sem,tpoints)
    areaplot(tpoints[3],log10.((out[3][4] ./ pars[36]) .^ pars[37]),label="mth1",title="HXT4",alpha=0.5)
    areaplot!(tpoints[3],log10.((out[3][2] ./ pars[40]) .^ pars[42]),label="mig1",alpha=0.5)
    areaplot!(tpoints[3],log10.((out[3][3] ./ pars[41]) .^ pars[43]),label="mig2",alpha=0.5)
    areaplot!(tpoints[3],log10.((out[3][5] ./ pars[38]) .^ pars[39]),label="std1",alpha=0.5,ylabel="log10(repression strength)")
    savefig("repression_midpoint_log10_$i.png")

    p1=areaplot(tpoints[3],(out[3][4] ./ pars[36]) .^ pars[37],label="mth1",ylabel="repression strength",title="HXT4",alpha=0.5,ylim=(0,1.2),xlabel="time")
    p2=areaplot(tpoints[3],(out[3][5] ./ pars[38]) .^ pars[39],label="std1",alpha=0.5,ylabel="repression strength",ylim=(0,1.2),xlabel="time")
    plot(p1,p2)
    savefig("repression_midpoint_$i.png")

    areaplot(tpoints[3],out[3][4],label="mth1",title="HXT4",alpha=0.5)
    areaplot!(tpoints[3],out[3][2],label="mig1",alpha=0.5)
    areaplot!(tpoints[3],out[3][3],label="mig2",alpha=0.5)
    areaplot!(tpoints[3],out[3][5],label="std1",alpha=0.5)
    savefig("repression_midpoint_$(i)_raw.png")
end

errors=[]
for i in 245:273
    pars = readdlm("int_g_midpoint_parameters_$i.txt")[:,1]
    push!(errors,rho_lens(pars))
end

ind = sortperm(errors)

rho_lens(readdlm("int_g_midpoint_parameters_$(ind[1]).txt")[:,1])

for i in ind[1:5]
    pars = readdlm("int_g_midpoint_parameters_$i.txt")[:,1]
    scatter(["dmig2g","kdmig2","mdmig2","r"],10 .^ pars[46:49],label="$i",alpha=1.0)
    scatter!(["dmig2g","kdmig2","mdmig2","r"],[10.0 ^ custombounds101[j][1] for j in 46:49],label="lower",alpha=0.5)
    scatter!(["dmig2g","kdmig2","mdmig2","r"],[10.0 ^ custombounds101[j][2] for j in 46:49],label="upper",alpha=0.5,legend=:topleft,yscale=:log10)
    savefig("mig2_params_$i.png")
end




plot(tpoints[1],int_glucose_interp[1](tpoints[1]))
plot!(tpoints[2],int_glucose_interp[2](tpoints[2]))
plot!(tpoints[3],int_glucose_interp[3](tpoints[3]))
ylabel!("int glucose")
xlabel!("time (hr)")
title!("WT")


plot(tpoints[10],int_glucose_interp[10](tpoints[10]))
plot!(tpoints[11],int_glucose_interp[11](tpoints[11]))
plot!(tpoints[12],int_glucose_interp[12](tpoints[12]),yscale=:log10)
ylabel!("int glucose")
xlabel!("time (hr)")
title!("mth1")

plot(tpoints[13],int_glucose_interp[13](tpoints[13]))
plot!(tpoints[14],int_glucose_interp[14](tpoints[14]))
plot!(tpoints[15],int_glucose_interp[15](tpoints[15]))
ylabel!("int glucose")
xlabel!("time (hr)")
title!("rgt2")

plot(tpoints[10],glucose_interp[10](tpoints[10]))
plot!(tpoints[10],int_glucose_interp[10](tpoints[10]))

plot(tpoints[11],glucose_interp[11](tpoints[11]))
plot!(tpoints[11],int_glucose_interp[11](tpoints[11]))

pars = readdlm("int_g_midpoint_parameters_769.txt")

pars = 10 .^ pars

problems =[]
glucose_interp = []
int_glucose_interp = []
count = 0
for h in unique(data[!,:strain])[1:6]
    for g in unique(data[!,:experiment_type])
        temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
        temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
        if !isempty(temp_data3)
            temp_data4 = temp_data1[!, [1,3]]
            temp_data6 = temp_data3[!, [1,3]]
            times = temp_data3[!,:time]
            fluor = temp_data4[!,:signal_value]
            count = count + 1
            if minimum(times) !== 0.0
                glucose = LinearInterpolation(vcat(0.0,times),vcat(0.0,temp_data6[1:length(times),:signal_value]),extrapolation_bc=Line()) #some glucose time series miss out the first time point/value! this is my hack
                int_glucose = intracellular_glucose(vcat(0.0,times),vcat(0.0,temp_data6[1:length(times),:signal_value]),pars[48 + count])
                push!(glucose_interp,glucose)
                push!(int_glucose_interp,int_glucose)
            else
                glucose = LinearInterpolation(times,temp_data6[1:length(times),:signal_value],extrapolation_bc=Line())
                int_glucose = intracellular_glucose(times,temp_data6[1:length(times),:signal_value],pars[48 + count])
                push!(glucose_interp,glucose)
                push!(int_glucose_interp,int_glucose)
            end

            if h =="HXT4_GFP"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))
            elseif h == "HXT4_GFP_snf3_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], snf3ko))
            elseif h == "HXT4_GFP_rgt2_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], rgt2ko))
            elseif h == "HXT4_GFP_std1_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], std1ko))
            elseif h == "HXT4_GFP_mig1_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mig1ko))
            elseif h == "HXT4_GFP_mth1_delta"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], mth1ko))
            end
        end
    end
end


using Plots
out_s=[]
out_m=[]
out=[]
#mkdir("output_$i")
for jj = 1:length(problems)
    if names[jj] ∈ ["HXT4_GFP_step_glucose_0p01_percent","HXT4_GFP_step_glucose_0p1_percent","HXT4_GFP_step_glucose_1p0_percent"]
             push!(out,ones(length(tpoints[jj])))
    else   
        sol = solve(problems[jj](pars[1:48]), TRBDF2(), saveat = tpoints[jj],tstops=tpoints[jj], verbose = false, abstol = 1e-12, reltol = 1e-12, dtmin = 1e-12,maxiters=1e6)
        #sol = solve(problems[jj](pars[1:48]), lsoda(), saveat = tpoints[jj], verbose = false, abstol = 1e-12, reltol = 1e-12, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])][1]
        push!(out,arr)
    end
end

plts = []
count = 0
gr(label="",grid=:on)
ys =[(0,4.5),(0,3.5),(0,15),(0,10),(0,3.5),(0,3),(0,4.5)]
for j in unique_experiment
    count+=1
    @show count
    if count ∈ [1,3,4,5,6,7]
        p1 = Plots.plot(tpoints[1+(count-1)*3:3+(count-1)*3],out[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",ylims=ys[count],linewidth=3.0)
        plot(p1,Plots.plot!(tpoints[1+(count-1)*3:3 .+ (count-1)*3],fluorescence[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",ylims=ys[count],linestyle=:dashdot))
        push!(plts,p1)
    end
end
display(plot(plts...,layout=(6,1),size=(500,1000)))
savefig("HXT4_midpoint_$(i)_check.png")


out_s=[]
out_m=[]
out=[]
#mkdir("output_$i")
for jj = 1:length(problems)
    if names[jj] ∈ ["HXT4_GFP_step_glucose_0p01_percent","HXT4_GFP_step_glucose_0p1_percent","HXT4_GFP_step_glucose_1p0_percent"]
             push!(out,ones(length(tpoints[jj])))
    else   
        sol = solve(problems[jj](pars[1:48]), TRBDF2(), saveat = tpoints[jj],tstops=tpoints[jj], verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
        #sol = solve(problems[jj](pars[1:48]), lsoda(), saveat = tpoints[jj], verbose = false, abstol = 1e-12, reltol = 1e-12, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])][2]
        push!(out,arr)
    end
end

plts = []
count = 0
gr(label="",grid=:on)
for j in unique_experiment
    count+=1
    @show count
    if count ∈ [1,3,4,5,6,7]
        p1 = Plots.plot(tpoints[1+(count-1)*3:3+(count-1)*3],out[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",linewidth=3.0)
        push!(plts,p1)
    end
end
display(plot(plts...,layout=(6,1),size=(500,1000)))
savefig("mig1_midpoint_$(i)_check.png")

out_s=[]
out_m=[]
out=[]
#mkdir("output_$i")
for jj = 1:length(problems)
    if names[jj] ∈ ["HXT4_GFP_step_glucose_0p01_percent","HXT4_GFP_step_glucose_0p1_percent","HXT4_GFP_step_glucose_1p0_percent"]
             push!(out,ones(length(tpoints[jj])))
    else   
        sol = solve(problems[jj](pars[1:48]), TRBDF2(), saveat = tpoints[jj],tstops=tpoints[jj], verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
        #sol = solve(problems[jj](pars[1:48]), lsoda(), saveat = tpoints[jj], verbose = false, abstol = 1e-12, reltol = 1e-12, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])][3]
        push!(out,arr)
    end
end

plts = []
count = 0
gr(label="",grid=:on)
for j in unique_experiment
    count+=1
    @show count
    if count ∈ [1,3,4,5,6,7]
        p1 = Plots.plot(tpoints[1+(count-1)*3:3+(count-1)*3],out[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",linewidth=3.0)
        push!(plts,p1)
    end
end
display(plot(plts...,layout=(6,1),size=(500,1000)))
savefig("mig2_midpoint_$(i)_check.png")

out_s=[]
out_m=[]
out=[]
#mkdir("output_$i")
for jj = 1:length(problems)
    if names[jj] ∈ ["HXT4_GFP_step_glucose_0p01_percent","HXT4_GFP_step_glucose_0p1_percent","HXT4_GFP_step_glucose_1p0_percent"]
             push!(out,ones(length(tpoints[jj])))
    else   
        sol = solve(problems[jj](pars[1:48]), TRBDF2(), saveat = tpoints[jj],tstops=tpoints[jj], verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
        #sol = solve(problems[jj](pars[1:48]), lsoda(), saveat = tpoints[jj], verbose = false, abstol = 1e-12, reltol = 1e-12, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])][4]
        push!(out,arr)
    end
end

plts = []
count = 0
gr(label="",grid=:on)
for j in unique_experiment
    count+=1
    @show count
    if count ∈ [1,3,4,5,6,7]
        p1 = Plots.plot(tpoints[1+(count-1)*3:3+(count-1)*3],out[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",linewidth=3.0)
        push!(plts,p1)
    end
end
display(plot(plts...,layout=(6,1),size=(500,1000)))
savefig("mth1_midpoint_$(i)_check.png")

out_s=[]
out_m=[]
out=[]
#mkdir("output_$i")
for jj = 1:length(problems)
    if names[jj] ∈ ["HXT4_GFP_step_glucose_0p01_percent","HXT4_GFP_step_glucose_0p1_percent","HXT4_GFP_step_glucose_1p0_percent"]
             push!(out,ones(length(tpoints[jj])))
    else   
        sol = solve(problems[jj](pars[1:48]), TRBDF2(), saveat = tpoints[jj],tstops=tpoints[jj], verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
        #sol = solve(problems[jj](pars[1:48]), lsoda(), saveat = tpoints[jj], verbose = false, abstol = 1e-12, reltol = 1e-12, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])][5]
        push!(out,arr)
    end
end

plts = []
count = 0
gr(label="",grid=:on)
for j in unique_experiment
    count+=1
    @show count
    if count ∈ [1,3,4,5,6,7]
        p1 = Plots.plot(tpoints[1+(count-1)*3:3+(count-1)*3],out[1+(count-1)*3:3 .+ (count-1)*3],title=unique_experiment[count],xlabel="time (hours)",linewidth=3.0)
        push!(plts,p1)
    end
end
display(plot(plts...,layout=(6,1),size=(500,1000)))
savefig("std1_midpoint_$(i)_check.png")


plot(tpoints[1],int_glucose_interp[1](tpoints[1]))
plot!(tpoints[2],int_glucose_interp[2](tpoints[2]))
plot!(tpoints[3],int_glucose_interp[3](tpoints[3]))
plot!(tpoints[4],int_glucose_interp[4](tpoints[4]))
plot!(tpoints[5],int_glucose_interp[5](tpoints[5]))
plot!(tpoints[6],int_glucose_interp[6](tpoints[6]),yscale=:linear)
ylabel!("int glucose")
xlabel!("time (hr)")
title!("WT")
savefig("int_g_wt.png")

plot(tpoints[7],int_glucose_interp[7](tpoints[7]))
plot!(tpoints[8],int_glucose_interp[8](tpoints[8]))
plot!(tpoints[9],int_glucose_interp[9](tpoints[9]),yscale=:log10,ylims=(10^-8,10^0))
ylabel!("int glucose")
xlabel!("time (hr)")
title!("mig1 delta")
savefig("int_g_mig1.png")


plot(tpoints[10],int_glucose_interp[10](tpoints[10]))
plot!(tpoints[11],int_glucose_interp[11](tpoints[11]))
plot!(tpoints[12],int_glucose_interp[12](tpoints[12]),yscale=:log10,ylims=(10^-8,10^0))
ylabel!("int glucose")
xlabel!("time (hr)")
title!("mth1 delta")
savefig("int_g_mth1.png")

plot(tpoints[13],int_glucose_interp[13](tpoints[13]))
plot!(tpoints[14],int_glucose_interp[14](tpoints[14]))
plot!(tpoints[15],int_glucose_interp[15](tpoints[15]),yscale=:log10,ylims=(10^-8,10^0))
ylabel!("int glucose")
xlabel!("time (hr)")
title!("rgt2 delta")
savefig("int_g_rgt2.png")


plot(tpoints[16],int_glucose_interp[16](tpoints[16]))
plot!(tpoints[17],int_glucose_interp[17](tpoints[17]))
plot!(tpoints[18],int_glucose_interp[18](tpoints[18]),yscale=:log10,ylims=(10^-8,10^0))
ylabel!("int glucose")
xlabel!("time (hr)")
title!("snf3 delta")
savefig("int_g_snf3.png")

plot(tpoints[19],int_glucose_interp[19](tpoints[19]))
plot!(tpoints[20],int_glucose_interp[20](tpoints[20]))
plot!(tpoints[21],int_glucose_interp[21](tpoints[21]),yscale=:log10,ylims=(10^-8,10^0))
ylabel!("int glucose")
xlabel!("time (hr)")
title!("std1 delta")
savefig("int_g_std1.png")


plot(tpoints[1],glucose_interp[1](tpoints[1]),label="")
plot!(tpoints[2],glucose_interp[2](tpoints[2]),label="")
plot!(tpoints[3],glucose_interp[3](tpoints[3]),label="")
plot!(tpoints[4],glucose_interp[4](tpoints[4]),label="")
plot!(tpoints[5],glucose_interp[5](tpoints[5]),label="")
plot!(tpoints[6],glucose_interp[6](tpoints[6]),yscale=:linear,label="",xticks=0:2.5:tpoints[6][end])
ylabel!("ext glucose")
xlabel!("time (hr)")
title!("WT")
savefig("ext_g_wt.png")

plot(tpoints[7],glucose_interp[7](tpoints[7]),label="")
plot!(tpoints[8],glucose_interp[8](tpoints[8]),label="")
plot!(tpoints[9],glucose_interp[9](tpoints[9]),yscale=:linear,label="",xticks=0:2.5:tpoints[9][end])
ylabel!("ext glucose")
xlabel!("time (hr)")
title!("mig1 delta")
savefig("ext_g_mig1.png")


plot(tpoints[10],glucose_interp[10](tpoints[10]),label="")
plot!(tpoints[11],glucose_interp[11](tpoints[11]),label="")
plot!(tpoints[12],glucose_interp[12](tpoints[12]),yscale=:linear,label="",xticks=0:2.5:tpoints[12][end])
ylabel!("ext glucose")
xlabel!("time (hr)")
title!("mth1 delta")
savefig("ext_g_mth1.png")

plot(tpoints[13],glucose_interp[13](tpoints[13]),label="")
plot!(tpoints[14],glucose_interp[14](tpoints[14]),label="")
plot!(tpoints[15],glucose_interp[15](tpoints[15]),yscale=:linear,label="",xticks=0:2.5:tpoints[15][end])
ylabel!("ext glucose")
xlabel!("time (hr)")
title!("rgt2 delta")
savefig("ext_g_rgt2.png")


plot(tpoints[16],glucose_interp[16](tpoints[16]),label="")
plot!(tpoints[17],glucose_interp[17](tpoints[17]),label="")
plot!(tpoints[18],glucose_interp[18](tpoints[18]),yscale=:linear,label="",xticks=0:2.5:tpoints[18][end])
ylabel!("ext glucose")
xlabel!("time (hr)")
title!("snf3 delta")
savefig("ext_g_snf3.png")

plot(tpoints[19],glucose_interp[19](tpoints[19]),label="")
plot!(tpoints[20],glucose_interp[20](tpoints[20]),label="")
plot!(tpoints[21],glucose_interp[21](tpoints[21]),yscale=:linear,label="",xticks=0:2.5:tpoints[21][end])
ylabel!("ext glucose")
xlabel!("time (hr)")
title!("std1 delta")
savefig("ext_g_std1.png")


parameter_names = [
    "k3", "thk", "ksnf1", "ksnf1std1", "nsnf1", "nsnf2", "dmth1", 
    "nmth1snf3", "nmth1rgt2", "thsnf3", "dmth1rgt2", "thmth1", 
    "kmth1mig1", "nmth1mig1", "kmth1mig2", "nmth1mig2", "thstd1", 
    "istd1", "nstd1", "nstd3", "thrgt2", "imig1", "kmig1snf1", 
    "theta2", "dmig2", "kmig2std1", "nmig2std1", "kmig2mth1", 
    "nmig2mth1", "dhxt4", "dhxt4max", "kdhxt4", "ndhxt4", "shxt4", 
    "khxt4mth1", "nhxt4mth1", "khxt4std1", "nhxt4std1", "khxt4mig1", 
    "khxt4mig2", "nhxt4mig1", "nhxt4mig2", "estd1max3", "ell", 
    "thmig2", "smig2", "kdmig2", "mdmig2", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", 
    "r9", "r10", "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", 
    "r19", "r20", "r21"
]

p = readdlm("int_g_midpoint_parameters_846.txt")[:,1]
using GlobalSensitivity
using QuasiMonteCarlo
using Plots
lb = [custombounds101[i][1] for i in 1:69]
ub = [custombounds101[i][2] for i in 1:69]
samples = 100000
sampler_choice = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(samples,lb,ub,sampler_choice)
@time sobol_sens = gsa(rho_batch, Sobol(order=[0,1,2]), A,B,batch=true)
# @time eFAST_sens = gsa(rho_batch, eFAST(),[[lb[i],ub[i]] for i in 1:47],samples=samples,batch=true)
# Create the bar plot
bar(sobol_sens.S1, 
    ylabel = "Sobol First order", 
    label = "", 
    xlabel = "parameters", 
    xticks=(1:length(parameter_names), parameter_names),
    xrotation=90,
    xtickfont=font(10),size=(1200,500))
savefig("first_sobol_sensitivity_global.png")
bar(sobol_sens.ST, 
    ylabel = "Sobol Total order", 
    label = "", 
    xlabel = "parameters", 
    xticks=(1:length(parameter_names), parameter_names),
    xrotation=90,
    xtickfont=font(10),size=(1200,500))
savefig("total_sobol_sensitivity_global.png")
heatmap(sobol_sens.S2, ylabel = "parameters",label="",xlabel="parameters",title="Sobol Second Order")
savefig("second_sobol_sensitivity_global.png")

lb = [max(p[i]-0.1,custombounds101[i][1]) for i in 1:69]
ub = [min(p[i]+0.1,custombounds101[i][2]) for i in 1:69]
samples = 100000
sampler_choice = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(samples,lb,ub,sampler_choice)
@time sobol_sens2 = gsa(rho_batch, Sobol(order=[0,1,2]), A,B,batch=true)
#@time eFAST_sens2 = gsa(rho_batch, eFAST(),[[lb[i],ub[i]] for i in 1:47],samples=samples,batch=true)

bar(sobol_sens2.S1, 
    ylabel = "Sobol First order", 
    label = "", 
    xlabel = "parameters", 
    xticks=(1:length(parameter_names), parameter_names),
    xrotation=90,
    xtickfont=font(10),size=(1200,500))
savefig("extended_first_sobol_sensitivity_around_particle_846.png")

bar(sobol_sens2.ST, 
    ylabel = "Sobol Total order", 
    label = "", 
    xlabel = "parameters", 
    xticks=(1:length(parameter_names), parameter_names),
    xrotation=90,
    xtickfont=font(10),size=(1200,500))
    savefig("extended_total_sobol_sensitivity_around_particle_846.png")
heatmap(sobol_sens2.S2, ylabel = "parameters",label="",xlabel="parameters",title="Sobol Second Order")
savefig("second_sobol_sensitivity_global_particles_846.png")
