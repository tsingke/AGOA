
## AGOA: Adversarial Game Optimization Algorithm

**Title**: Adversarial Game Optimization Algorithm: A game-theoretic metaheuristic for efficient global optimization and engineering application

```
 Authors：Conglin Li， Qingke Zhang*， Junqing Li, Sichen Tao and Diego Oliva
```
> 1. School of Information Science and Engineering, Shandong Normal University, Jinan 250358, China
> 
> 2. School of Information Science and Engineering, Yunnan Normal University,  Yunan 650500, China
> 
> 3. Faculty of Engineering, University of Toyama, Toyama-shi 930-8555, Japan
> 
> 4. Depto. de Ingeniería Electro-Fótonca, Universidad de Guadalajara, CUCEI, 44430, Guadalajara, México

> Corresponding Author: **Qingke Zhang** ， Email: tsingke@sdnu.edu.cn ， Tel :  +86-13953128163

This paper is being considered for submission to the international journal **Information Sciences** (JCR Q1）

### 1. Introduction

This paper presents the Adversarial Game Optimization Algorithm (AGOA), an innovative metaheuristic framework inspired by adversarial game dynamics in game theory, designed to overcome challenges such as premature convergence and insufficient global exploration in high-dimensional and multimodal optimization problems. By introducing a dynamic role-based population partitioning strategy that seamlessly integrates elite guidance, adaptive ordinary exploration, and reverse feedback from underperforming individuals, AGOA achieves a robust and adaptive balance between global exploration and local exploitation. Extensive experiments on CEC 2017 and CEC 2022 benchmarks, along with a complex UAV 3D path planning application, demonstrate AGOA’s superior convergence accuracy, search efficiency, and solution robustness compared to 75 state-of-the-art algorithms. Moreover, a rigorous theoretical analysis establishes AGOA’s convergence to the global optimum under practical assumptions, further validating its methodological soundness and highlighting its strong potential for solving challenging real-world engineering optimization tasks. The source code of AGOA is openly available at https://github.com/tsingke/AGOA.


### 2. Schematic Diagram of AGOA


<img width="600" alt="image" src="https://github.com/user-attachments/assets/f5370719-974c-4e50-8831-44359a794806" />

### 3. The pseudocode of AGOA optimizer
<img width="670" alt="image" src="https://github.com/user-attachments/assets/e36b2013-7af4-4049-b9e8-92bca89a3569" />


### 4. The MATLAB code of AGOA
```MATLAB
% =========================================================================
%  AGOA: Adversarial Game Optimization Algorithm
%  Copyright (c) 2025, Qingke Zhang @ SDNU CILab
%  This code is released for academic and research use.
%  Please cite the related paper when using or modifying.
% =========================================================================
function [gbestX, gbestfitness, gbesthistory] = AGOA(popsize, dimension, xmax, xmin, maxiter, Func, FuncId)

FEs = 0;
MaxFEs = popsize * maxiter;
Fitness = Func;

% Initialize population
x = xmin + (xmax - xmin) * rand(popsize, dimension);
fitness = zeros(1, popsize);
gbestfitness = inf;
gbesthistory = zeros(1, MaxFEs);

% Evaluate initial population
for i = 1:popsize
    fitness(i) = Fitness(x(i, :)', FuncId);
    FEs = FEs + 1;
    if fitness(i) < gbestfitness
        gbestfitness = fitness(i);
        gbestX = x(i, :);
    end
    gbesthistory(FEs) = gbestfitness;
end

% Initialize parameters
Er = 0.1; % Elite reinforcement factor
Dc = 0.1; % Defensive counterattack factor
Co = 0.1; % Cooperative probability
F = 0.7;  % Base scaling factor
chaos_factor = 0.1; % Chaos factor for adaptive scaling
Dr = 0.1; % Diversity threshold
stagnation_count = 0;
replay_buffer = struct('positions', {}, 'fitness', {});

while FEs <= MaxFEs

    % Sort and assign roles
    [~, sorted_indices] = sort(fitness);
    elite_count = ceil(popsize * 0.2);
    normal_count = ceil(popsize * 0.6);
    worst_count = popsize - elite_count - normal_count;

    elite_indices = sorted_indices(1:elite_count);
    normal_indices = sorted_indices(elite_count+1 : elite_count+normal_count);
    worst_indices = sorted_indices(elite_count+normal_count+1 : end);

    % Iterate through individuals
    for i = 1:popsize
        % Adaptive chaos-based scaling
        chaos_value = chaos_factor * rand + (1 - chaos_factor) * sin(rand * pi);
        F_adaptive = F * chaos_value;

        % Randomly select individuals from each role
        x1 = x(elite_indices(randi(length(elite_indices))), :);
        x2 = x(normal_indices(randi(length(normal_indices))), :);
        x3 = x(normal_indices(randi(length(normal_indices))), :);
        x4 = x(worst_indices(randi(length(worst_indices))), :);

        % Generate mutant vector
        mutant = Er * x1 + F_adaptive * ((x2 - x3) - Dc * x4);
        mutant_fitness = Fitness(mutant', FuncId);
        FEs = FEs + 1;

        % Evaluate and decide acceptance
        if mutant_fitness < gbestfitness
            trial = mutant;
            trial_fitness = mutant_fitness;
        else
            CR_adaptive = Co * chaos_value;
            cross_points = rand(1, dimension) < CR_adaptive;
            if ~any(cross_points)
                cross_points(randi(dimension)) = true;
            end
            trial = x(i, :);
            trial(cross_points) = mutant(cross_points);
            trial_fitness = Fitness(trial', FuncId);
            FEs = FEs + 1;
        end

        % Accept or reject trial vector
        if trial_fitness < fitness(i)
            x(i, :) = trial;
            fitness(i) = trial_fitness;
            stagnation_count = 0;
        else
            stagnation_count = stagnation_count + 1;
        end

        % Update global best
        if trial_fitness < gbestfitness
            gbestfitness = trial_fitness;
            gbestX = trial;
        end
        gbesthistory(FEs) = gbestfitness;

        % Optional progress output
        if mod(FEs, MaxFEs / 10) == 0
            fprintf('AGOA evaluation %d, best fitness = %.4e\n', FEs, gbestfitness);
        end
    end

    % Replay buffer-based parameter adaptation
    if ~isempty(replay_buffer)
        replay_count = min(10, length(replay_buffer));
        for j = 1:replay_count
            idx = randi(length(replay_buffer));
            diff = gbestfitness - min(replay_buffer(idx).fitness);
            if diff > 0
                Er = Er + 0.1 * diff;
                Dc = Dc - 0.1 * diff;
            else
                Er = Er - 0.1 * diff;
                Dc = Dc + 0.1 * diff;
            end
        end
        Er = max(min(Er, 1), 0);
        Dc = max(min(Dc, 1), 0);
    end

    % Update replay buffer
    replay_buffer(end + 1).positions = x;
    replay_buffer(end).fitness = fitness;
    if length(replay_buffer) > 100
        replay_buffer(1) = [];
    end

    % Diversity monitoring and restart worst individuals if needed
    population_diversity = mean(std(x));
    if population_diversity < Dr || stagnation_count > 100
        for k = 1:worst_count
            new_individual = xmin + (xmax - xmin) * rand(1, dimension);
            x(worst_indices(k), :) = new_individual;
            fitness(worst_indices(k)) = Fitness(new_individual', FuncId);
            FEs = FEs + 1;
        end
        stagnation_count = 0;
    end

    if FEs >= MaxFEs
        break;
    end
end

% Fill history if terminated early
if FEs < MaxFEs
    gbesthistory(FEs+1 : MaxFEs) = gbestfitness;
else
    gbesthistory(MaxFEs+1:end) = [];
end

end

```
## 5. Acknowledgements

**We would like to express our sincere gratitude to the anonymous reviewers for taking the time to review our paper.** This work is supported by the National Natural Science Foundation of China (No. 62006144) and the Taishan Scholar Project of Shandong, China (No.ts20190924).



