
## AGOA: Adversarial Game Optimization Algorithm

**Title**: Adversarial Game Optimization Algorithm: A novel metaheuristic optimization approach for complex optimization problem

```
 Authors：Conglin Li, Qingke Zhang*, Shuzhao Pang，Wenliang Chen， Huaxiang Zhang
```
*  School of Information Science and Engineering, Shandong Normal University, Jinan 250358, China

> Corresponding Author: **Qingke Zhang**

> Email: tsingke@sdnu.edu.cn

> Tel :  +86-13953128163


### 1. Introduction

The Adversarial Game Optimization Algorithm (AGOA) presents an innovative framework that incorporates game-theoretic adversarial dynamics into metaheuristic optimization. By assigning individuals to elite, ordinary, and worst roles, AGOA adaptively balances global exploration with local refinement, effectively enhancing search efficiency. The integration of chaotic adaptive parameter adjustment and diversity-driven restart strategies helps maintain population diversity and mitigate premature convergence. Additionally, the game experience replay mechanism enables self-adaptive fine-tuning of parameters based on historical performance, supporting stable and accelerated convergence. This carefully designed combination of adversarial interactions, adaptive learning, and diversity control makes AGOA a promising and versatile approach for addressing complex high-dimensional optimization challenges.


### 2. Schematic Diagram of AGOA

<img width="450" alt="image" src="https://github.com/user-attachments/assets/b65c6a61-82a0-4179-8c80-e6f562cc26ae" />


### 3. The pseudocode of AGOA optimizer
<img width="470" alt="image" src="https://github.com/user-attachments/assets/e36b2013-7af4-4049-b9e8-92bca89a3569" />


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



