using MKL
using Base.Threads
using StatsBase, Random
using Distributions
using DelimitedFiles
using LinearAlgebra
using OrdinaryDiffEq
using CSV, DataFrames
using Interpolations
#data = CSV.read("HXTs_for_fitting.csv", DataFrame)
atol = 1e-6
rtol=1e-9
data = CSV.read("mean_HXTs_for_fitting.csv",DataFrame)
function wt(params)
    params
end
#genotype functions alter parameters to simulate mutants.

function mth1ko(params)
    pars = copy(params)
    #pars = 10 .^ (pars)
    pars[12] = -(pars[17]+pars[37])
    return collect(pars)
end

function std1ko(params)
    pars = copy(params)
    #pars = 10 .^ (pars)
    pars[18] = 0.0
    return collect(pars)
end	

function mig2ko(params)
    pars = copy(params)
    #pars = 10 .^ (pars)
    pars[46] = 0.0
    return collect(pars)
end	

function mig1ko(params)
    pars = copy(params)
    #pars = 10 .^ (pars)
    pars[22] = 0.0 #imig1
    return collect(pars)
end	

function rgt2ko(params)
    pars = copy(params)
    #pars = 10 .^ (pars)
    pars[11] = 0.0 #dmth1rgt2
    pars[21] = 0.0 #thrgt2#added by Luis 20201109
    return collect(pars)
end

function snf3ko(params)
    pars = copy(params)
    #pars = 10 .^ (pars)
    pars[10] = 0.0 #thsnf3
    pars[43] = 0.0 #estd1max3
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
        sol = solve(prob, TRBDF2(), save_everystep = false, verbose = false, abstol = atol, reltol = rtol,dtmin=1e-14)
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
        sol = solve(prob, TRBDF2(), save_everystep = false, verbose = false, abstol = atol, reltol = rtol,dtmin=1e-14)
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

            # if istd1 == 0.0
            #     std1tot = 0.0
            # else
            #     std1tot = kmig2std1 + thstd1
            # end
            std1tot = kmig2std1 + thstd1
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
            if nhxt4mig2 == 1e10 && nhxt4std1 == 1e10
                dydt[1] = +(shxt4 / (1 + (Mth1 / khxt4mth1)^nhxt4mth1 + (Mig1 / khxt4mig1)^nhxt4mig1)) - dhxt4 * Hxt4 - (dhxt4max * Hxt4) / (1 + (ig / kdhxt4)^ndhxt4) 
            elseif nhxt4mig2 == 1e10
                dydt[1] = +(shxt4 / (1 + (Mth1 / khxt4mth1)^nhxt4mth1 + (Std1 / khxt4std1)^nhxt4std1 + (Mig1 / khxt4mig1)^nhxt4mig1)) - dhxt4 * Hxt4 - (dhxt4max * Hxt4) / (1 + (ig / kdhxt4)^ndhxt4) 
            elseif nhxt4std1 == 1e10
                dydt[1] = +(shxt4 / (1 + (Mth1 / khxt4mth1)^nhxt4mth1 + (Mig1 / khxt4mig1)^nhxt4mig1 + (Mig2 / khxt4mig2)^nhxt4mig2)) - dhxt4 * Hxt4 - (dhxt4max * Hxt4) / (1 + (ig / kdhxt4)^ndhxt4) 
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
for h in unique(data[!,:strain])
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

            if h =="HXT6_GFP"
                push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))
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
        sol = solve(prob(pars), TRBDF2(), saveat = times,tstops=times, verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])]
        #lsqval = mean((arr[1] - datamean).^2 ./ datasem.^2) #+ 1*(1/(maximum(sol.u[2]/pars[43])) + 1/(maximum(sol.u[3]/pars[44])) + 1/(maximum(sol.u[4]/pars[38])) + 1/(maximum(sol.u[5]/pars[40])))
        lsqval = mean(((arr[1] - datamean).^2 ./ mean(datamean)))
    catch
        lsqval = 1e20
    end
end

function trysolvet(prob, pars, datamean, datasem,times) # main function to solve a system with two solvers and return a large value if both fail.
    try
        sol = solve(prob(pars), TRBDF2(), saveat = times,tstops=times, verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])]
        lsqval = arr
    catch
        lsqval = 1e20
    end
end

function trysolve2(prob, pars, datamean, datasem,times) # main function to solve a system with two solvers and return a large value if both fail.
    try
        sol = solve(prob(pars), TRBDF2(), saveat = times,tstops=times, verbose = false, abstol = 1e-6, reltol = 1e-6, dtmin = 1e-12,maxiters=1e6)
        arr = [[j[i] for j in sol.u] for i = 1:length(sol.u[1])]
        lsqval = arr[1]
    catch
        lsqval = 1e20
    end
end

function rho_lens15(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars;fixed_p[43:48];fixed_p[49:52]]

    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end    
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens14(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:10];fixed_p[40];pars[11];0.0;fixed_p[43:48];fixed_p[49:52]]
    pars[42] = 1e10 #mig2 - should be 40?
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens13(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:9];10.0;pars[10];1.0;pars[11];fixed_p[43:48];fixed_p[49:52]] #mig1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens12(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:9];10.0;fixed_p[40];1.0;1.0;fixed_p[43:48];fixed_p[49:52]]
    pars[42] = 1e10 #mig1, mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens11(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;pars[8:11];fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens10(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;pars[8];fixed_p[40];pars[9];1.0;fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1
    pars[42] = 1e10 #mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens9(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;10.0;pars[8];0.0;pars[9];fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1, mig1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens8(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;10.0;fixed_p[40];0.0;1.0;fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1, mig1
    pars[42] = 1e10 #mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
      for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens7(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:11];fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens6(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:8];fixed_p[40];pars[9];1.0;fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[42] = 1e10 #mig2
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens5(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:7];10.0;pars[8];0.0;pars[9];fixed_p[43:48];fixed_p[49:52]]#mth1,mig1
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens4(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:7];10.0;fixed_p[40];0.0;0.0;fixed_p[43:48];fixed_p[49:52]]#mth1,mig1
    pars[42] = 1e10 #mig2
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens3(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;fixed_p[38];0.0;pars[6:9];fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[38] = 1e10 #std1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens2(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;1.0;fixed_p[38];1.0;pars[6];fixed_p[40];pars[7];1.0;fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[38] = 1e10 #std1
    pars[42] = 1e10 #mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
end

function rho_lens1(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;fixed_p[38];1.0;10.0;pars[6];0.0;pars[7];fixed_p[43:48];fixed_p[49:52]]#mth1, mig1
    pars[38] = 1e10 #std1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
    errors =  [i[1] for i in out]
    errors=errors[findall(errors .< 1e20)]
    if length(errors) < 4
        E = mean(errors) + 4.0 - length(errors) 
    else
        E = mean(errors .* [3.0,1.0,1.0,1.0]) + 4.0 - length(errors)
    end
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
(-1.5546342970446156,-0.3886585742611539), # dhxt4 30, 1
(-1.4346229631313553,-0.3586557407828388), # dhxt4max 31, 2
(1.9051015841091236,4.0), # kdhxt4 32, 3
(log10(3.844), log10(5.0)), # ndhxt4 33, 4
(-2.0,2.0), # shxt4 34, 5
(-10.0,log10(3.0)), #th khxt4mth1 35, 6
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
(log10(1.0),log10(5.0))] #r4 52

custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds1 = deleteat!(custombounds, [6,7,8,9,10,12])  # [0001]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds2 = deleteat!(custombounds, [6,7,8,9,11,13])  # [0010]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds3 = deleteat!(custombounds, [6,7,8,9])  # [0011]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds4 = deleteat!(custombounds, [6,7,10,11,12,13])  # [0100]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds5 = deleteat!(custombounds, [6,7,10,12])  # [0101]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds6 = deleteat!(custombounds, [6,7,11,13])  # [0110]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds7 = deleteat!(custombounds, [6,7])  # [0111]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds8 = deleteat!(custombounds, [8,9,10,11,12,13])  # [1000]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds9 = deleteat!(custombounds, [8,9,10,12])  # [1001]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds10 = deleteat!(custombounds, [8,9,11,13])  # [1010]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds11 = deleteat!(custombounds, [8,9])  # [1011]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds12 = deleteat!(custombounds, [10,11,12,13])  # [1100]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds13 = deleteat!(custombounds, [10,12])  # [1101]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds14 = deleteat!(custombounds, [11,13])  # [1110]
custombounds = copy(custombounds101)
deleteat!(custombounds, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,43,44,45,46,47,48])
custombounds15 = custombounds

model_lens1 = [Uniform(p[1], p[2]) for p in custombounds1]
model_lens2 = [Uniform(p[1], p[2]) for p in custombounds2]
model_lens3 = [Uniform(p[1], p[2]) for p in custombounds3]
model_lens4 = [Uniform(p[1], p[2]) for p in custombounds4]
model_lens5 = [Uniform(p[1], p[2]) for p in custombounds5]
model_lens6 = [Uniform(p[1], p[2]) for p in custombounds6]
model_lens7 = [Uniform(p[1], p[2]) for p in custombounds7]
model_lens8 = [Uniform(p[1], p[2]) for p in custombounds8]
model_lens9 = [Uniform(p[1], p[2]) for p in custombounds9]
model_lens10 = [Uniform(p[1], p[2]) for p in custombounds10]
model_lens11 = [Uniform(p[1], p[2]) for p in custombounds11]
model_lens12 = [Uniform(p[1], p[2]) for p in custombounds12]
model_lens13 = [Uniform(p[1], p[2]) for p in custombounds13]
model_lens14 = [Uniform(p[1], p[2]) for p in custombounds14] #Mig2 K0
model_lens15 = [Uniform(p[1], p[2]) for p in custombounds15]

np = 20000

include("types.jl")
include("functions.jl")
include("updated_abc_model_comparison_threads.jl")


for ii = 1:10
    apmc_output = APMC(np, [model_lens12,model_lens13,model_lens14,model_lens15], [rho_lens12,rho_lens13,rho_lens14,rho_lens15], perturb="Cauchy",n=0.5)

    writedlm("hxt6_955_shrunk/p_$(ii).txt", apmc_output.p)
    writedlm("hxt6_955_shrunk/e_$(ii).txt", apmc_output.epsilon)
    for jj in 1:4
        writedlm("hxt6_955_shrunk/pts_$(ii)_$jj.txt",apmc_output.pts[jj,end])
        writedlm("hxt6_955_shrunk/wts_$(ii)_$jj.txt",apmc_output.wts[jj,end])
    end
end

using Plots
using DelimitedFiles
using StatsBase

final_p6 = zeros(4,10)
final_e6 = zeros(10)

for i = [6]
    for j = 1:10
        p = readdlm("hxt$(i)_955_shrunk/p_$j.txt")
        eval(Meta.parse("final_p$i"))[:,j] = p[:,end]
        e = readdlm("hxt$(i)_955_shrunk/e_$j.txt")
        eval(Meta.parse("final_e$i"))[j] = e[end]
    end
end

using Measures
i=6
p1=heatmap(eval(Meta.parse("final_p$i")),yticks=(1:4,[12,13,14,15]),xlabel="repeat",ylabel="model",colorbar_title="final probability",title="",clim=(0,1))
p2=bar(eval(Meta.parse("final_e$i")),xlabel="repeat",label="",ylabel="final error",title="")
p3=bar(mean(eval(Meta.parse("final_p$i")),dims=2),xticks = (1:4,[12,13,14,15]),xlabel="model",ylabel="average final probability",title="HXT$i",ylim=(0,1),label="")
plot(p1,p3,p2,layout=(1,3),size=(1200,400),margins=5mm)
savefig("heatmap_final_probabilities_hxt_rejigged_$(i)_shrunk.png")

using Plots
# ii=findmin(final_e1)[2]
# win=findmax(final_p1[:,ii])[2]
# win_pts = readdlm("hxt6_955_shrunk_2k/pts_$(ii)_$win.txt")[:,end]

win=findmax(mean(final_p6,dims=2))[2][1]
is=findall(final_p6[win,:] .> 0.5)
ii = findmin(final_e6[is])[2]
win_pts = readdlm("hxt6_955_shrunk/pts_$(ii)_$win.txt")[:,end]

function model15(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars;fixed_p[43:48];fixed_p[49:52]]

    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end    
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)
end

function model14(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:10];fixed_p[40];pars[11];0.0;fixed_p[43:48];fixed_p[49:52]]
    pars[42] = 1e10 #mig2 - should be 40?
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model13(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:9];10.0;pars[10];1.0;pars[11];fixed_p[43:48];fixed_p[49:52]] #mig1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model12(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:9];10.0;fixed_p[40];1.0;1.0;fixed_p[43:48];fixed_p[49:52]]
    pars[42] = 1e10 #mig1, mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model11(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;pars[8:11];fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model10(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;pars[8];fixed_p[40];pars[9];1.0;fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1
    pars[42] = 1e10 #mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model9(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;10.0;pars[8];0.0;pars[9];fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1, mig1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model8(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:7];fixed_p[37];0.0;10.0;fixed_p[40];0.0;1.0;fixed_p[43:48];fixed_p[49:52]]
    pars[38] = 1e10 #std1, mig1
    pars[42] = 1e10 #mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
      for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model7(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:11];fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model6(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:8];fixed_p[40];pars[9];1.0;fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[42] = 1e10 #mig2
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model5(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:7];10.0;pars[8];0.0;pars[9];fixed_p[43:48];fixed_p[49:52]]#mth1,mig1
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model4(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;pars[6:7];10.0;fixed_p[40];0.0;0.0;fixed_p[43:48];fixed_p[49:52]]#mth1,mig1
    pars[42] = 1e10 #mig2
    pars[35] = -10^10
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model3(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;fixed_p[38];0.0;pars[6:9];fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[38] = 1e10 #std1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model2(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;1.0;fixed_p[38];1.0;pars[6];fixed_p[40];pars[7];1.0;fixed_p[43:48];fixed_p[49:52]]#mth1
    pars[38] = 1e10 #std1
    pars[42] = 1e10 #mig2
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

function model1(d2)
    pars = copy(d2)
    fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
    pars = 10 .^ [fixed_p[1:29];pars[1:5];10.0;0.0;fixed_p[38];1.0;10.0;pars[6];0.0;pars[7];fixed_p[43:48];fixed_p[49:52]]#mth1, mig1
    pars[38] = 1e10 #std1
    problems =[]
    glucose_interp = []
    int_glucose_interp = []
    count = 0
    for h in unique(data[!,:strain])
        for g in unique(data[!,:experiment_type])
            temp_data1 = data[coalesce.(data.signal_type .== "fluorescence" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            temp_data3 = data[coalesce.(data.signal_type .== "glucose_proxy" .&& data.strain .== h .&& data.experiment_type .== g, false), [1,3,5]]
            if !isempty(temp_data3)
                temp_data4 = temp_data1[!, [1,3]]
                temp_data6 = temp_data3[!, [1,3]]
                times = temp_data3[!,:time]
                fluor = temp_data4[!,:signal_value]
                if h =="HXT6_GFP"
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
                    push!(problems,makeproblem1(int_glucose,glucose, (minimum(times),maximum(times)), fluor[1], wt))

                end
            end
        end
    end
    out = trysolve2.(problems, [pars[1:48]], fluorescence,fluorescence_sem,tpoints)

end

win = win + 11
sols=eval(Meta.parse("model$win(win_pts)"))

gr(label="",xlabel="time (hours)",ylabel="fluorescence")
p1=plot(tpoints[1],sols[1])
p1=plot!(tpoints[1],fluorescence[1],ribbon=fluorescence_sem[1])
p2=plot(tpoints[2],sols[2])
p2=plot!(tpoints[2],fluorescence[2],ribbon=fluorescence_sem[2])
p3=plot(tpoints[3],sols[3])
p3=plot!(tpoints[3],fluorescence[3],ribbon=fluorescence_sem[3])
p4=plot(tpoints[4],sols[4])
p4=plot!(tpoints[4],fluorescence[4],ribbon=fluorescence_sem[4])
plot(p1,p2,p3,p4,layout=(2,2))
savefig("winning_model_$(win)_sample_solution_hxt6_955_shrunk.png")

# bar(apmc_output.p[:,end],xlabel="model",ylabel="final probability")
# savefig("final_model_probabilities_HXT6.png")

#sols = trysolve2.(problems, [pars], fluorescence,fluorescence_sem,tpoints)

final_p6 = zeros(4,10)
final_e6 = zeros(10)

for j = 1:10
    p = readdlm("hxt6_955_shrunk/p_$j.txt")
    eval(Meta.parse("final_p6"))[:,j] = p[:,end]
    e = readdlm("hxt6_955_shrunk/e_$j.txt")
    eval(Meta.parse("final_e6"))[j] = e[end]
end
writedlm("final_p6_shrunk.txt",final_p6)
writedlm("final_e6_shrunk.txt",final_e6)

for ii = 1:10, win in [1,2,3,4]
    #win=findmax(final_p1[:,ii])[2]
    if final_p6[win,ii]>0
        if win==1
            pars = readdlm("hxt6_955_shrunk/pts_$(ii)_$win.txt")[:,1]
            fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
            pars = 10 .^ [fixed_p[1:29];pars[1:9];10.0;fixed_p[40];1.0;1.0;fixed_p[43:48];fixed_p[49:52]]
            pars[42] = 1e10 #mig1, mig2
        elseif win==2
            pars = readdlm("hxt6_955_shrunk/pts_$(ii)_$win.txt")[:,1]
            fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
            pars = 10 .^ [fixed_p[1:29];pars[1:9];10.0;pars[10];1.0;pars[11];fixed_p[43:48];fixed_p[49:52]] #mig1
        elseif win==3
            pars = readdlm("hxt6_955_shrunk/pts_$(ii)_$win.txt")[:,1]
            fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
            pars = 10 .^ [fixed_p[1:29];pars[1:10];fixed_p[40];pars[11];0.0;fixed_p[43:48];fixed_p[49:52]]
            pars[42] = 1e10 #mig2 - should be 40?
        elseif win==4
            pars = readdlm("hxt6_955_shrunk/pts_$(ii)_$win.txt")[:,1]
            fixed_p = readdlm("potential_particles/int_g_midpoint_parameters_955.txt")[:,1]
            pars = 10 .^ [fixed_p[1:29];pars;fixed_p[43:48];fixed_p[49:52]]
        end
        win = win+11

        writedlm("hxt_6_best_particle_model_$(win)_run_$(ii)_shrunk.txt",pars)
    end
end