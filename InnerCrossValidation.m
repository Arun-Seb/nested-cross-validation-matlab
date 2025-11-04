%% Nested Cross-Validation for Optimal Feature Selection
% This script implements nested cross-validation to find the best feature
% combination for classification tasks. The outer loop performs Leave-One-Out
% Cross-Validation (LOOCV) across patients, while the inner loop performs
% 10-fold cross-validation for feature selection.
%
% INPUTS (assumes these variables exist in workspace):
%   X - Cell array of feature matrices {NUM_PATIENTS x 1}, each cell is (samples x features)
%   T - Cell array of target matrices {NUM_PATIENTS x 1}, each cell is (samples x classes), one-hot encoded
%   PSOC_MANUAL - Cell array {NUM_PATIENTS x 1} of manual classifications ('T' or 'N')
% Sebastian A, Cistulli PA, Cohen G, de Chazal P. Association of snoring characteristics with predominant
% site of collapse of upper airway in obstructive sleep apnea patients. Sleep. 2021 Dec 10;44(12):
% zsab176. doi: 10.1093/sleep/zsab176. PMID: 34270768.
% Author: [Arun Sebastian]
% Date: [20/10/2023]
% License: MIT

%% Configuration
CLASSIFIER_METHOD = 'LDA'; % Options: 'LDA' or 'LR' (can be replaced with any classifier)
NUM_INNER_FOLDS = 10; % Number of folds for inner cross-validation
PREDOMINANCE_THRESHOLD = 0.6; % Threshold for class predominance classification

NUM_PATIENTS = length(X);
num_features = size(X{1}, 2);

fprintf('Starting nested cross-validation on %d patients with %d features...\n\n', NUM_PATIENTS, num_features);

%% ============================================================================
%% NESTED CROSS-VALIDATION FRAMEWORK
%% ============================================================================
% 
% OUTER LOOP (LOOCV): For each patient
%   - Hold out one patient as test set
%   - Use remaining patients as training set
%   
%   INNER LOOP (10-Fold CV): For feature selection
%     - Try adding each candidate feature
%     - Evaluate using 10-fold cross-validation
%     - Select feature that maximizes validation accuracy
%     - Repeat until all features evaluated
%   
%   - Train final model with optimal features on outer training set
%   - Evaluat/e on held-out test patient/
%
%% ============================================================================

% Preallocate results storage
inner_cv_accuracy = zeros(NUM_PATIENTS, num_features); % Accuracy for each feature count
inner_cv_selected_features = zeros(NUM_PATIENTS, num_features); % Selected feature indices
optimal_feature_count = zeros(NUM_PATIENTS, 1); % Optimal number of features per patient

T_train_outer = cell(NUM_PATIENTS, 1);
T_test_outer = cell(NUM_PATIENTS, 1);
O_train_outer = cell(NUM_PATIENTS, 1);
O_test_outer = cell(NUM_PATIENTS, 1);
PSOC_PREDICTED = cell(NUM_PATIENTS, 1);

%% OUTER LOOP: Leave-One-Out Cross-Validation (LOOCV)
for patient_idx = 1:NUM_PATIENTS
    fprintf('=== Patient %d/%d ===\n', patient_idx, NUM_PATIENTS);
    
    % -------------------------------------------------------------------------
    % STEP 1: Split data - Outer CV (LOOCV)
    % -------------------------------------------------------------------------
    train_patients = setdiff(1:NUM_PATIENTS, patient_idx); % All except current patient
    test_patient = patient_idx;
    
    % Concatenate training data from all training patients
    X_train = cat(1, X{train_patients});
    T_train_outer{patient_idx} = cat(1, T{train_patients});
    num_train_samples = size(X_train, 1);
    
    % Test data (single patient)
    X_test = cat(1, X{test_patient});
    T_test_outer{patient_idx} = cat(1, T{test_patient});
    num_test_samples = size(X_test, 1);
    
    % -------------------------------------------------------------------------
    % STEP 2: Sequential Forward Feature Selection using Inner CV
    % -------------------------------------------------------------------------
    selected_features = []; % Accumulate best features sequentially
    
    for feature_count = 1:num_features
        fprintf('  Evaluating feature count: %d/%d\n', feature_count, num_features);
        
        feature_accuracies = zeros(num_features, 1); % Store accuracy for each candidate feature
        
        % Try adding each remaining feature
        for candidate_feature = 1:num_features
            
            % Build current feature set
            if feature_count == 1
                % First iteration: evaluate each feature individually
                current_features = candidate_feature;
            else
                % Skip if feature already selected
                if ismember(candidate_feature, selected_features)
                    feature_accuracies(candidate_feature) = -inf; % Mark as invalid
                    continue;
                end
                % Add candidate to previously selected features
                current_features = [selected_features, candidate_feature];
            end
            
            X_train_subset = X_train(:, current_features);
            
            % -----------------------------------------------------------------
            % INNER LOOP: 10-Fold Cross-Validation
            % -----------------------------------------------------------------
            % Purpose: Evaluate how well this feature combination generalizes
            
            % Prepare data for inner CV
            num_all_samples = size(X_train_subset, 1);
            shuffle_idx = randperm(num_all_samples); % Randomize order
            
            % Round down to nearest multiple of NUM_INNER_FOLDS for equal folds
            num_cv_samples = floor(num_all_samples / NUM_INNER_FOLDS) * NUM_INNER_FOLDS;
            shuffle_idx = shuffle_idx(1:num_cv_samples);
            
            X_train_shuffled = X_train_subset(shuffle_idx, :);
            T_train_shuffled = T_train_outer{patient_idx}(shuffle_idx, :);
            
            % Split into NUM_INNER_FOLDS folds (interleaved sampling)
            X_folds = cell(NUM_INNER_FOLDS, 1);
            T_folds = cell(NUM_INNER_FOLDS, 1);
            for fold = 1:NUM_INNER_FOLDS
                fold_indices = fold:NUM_INNER_FOLDS:num_cv_samples; % Every Nth sample
                X_folds{fold} = X_train_shuffled(fold_indices, :);
                T_folds{fold} = T_train_shuffled(fold_indices, :);
            end
            
            % Perform K-fold cross-validation
            O_folds = cell(NUM_INNER_FOLDS, 1); % Store predictions for each fold
            
            for fold = 1:NUM_INNER_FOLDS
                % Split: train on K-1 folds, validate on 1 fold
                train_folds = setdiff(1:NUM_INNER_FOLDS, fold);
                val_fold = fold;
                
                X_train_inner = cat(1, X_folds{train_folds});
                T_train_inner = cat(1, T_folds{train_folds});
                num_train_inner = size(X_train_inner, 1);
                
                X_val = cat(1, X_folds{val_fold});
                T_val = cat(1, T_folds{val_fold});
                num_val = size(X_val, 1);
                
                % ---------------------------------------------------------
                % CLASSIFIER TRAINING (ANY CLASSIFIER CAN BE PLUGGED IN HERE)
                % ---------------------------------------------------------
                % Replace train_classifier() with your preferred algorithm:
                % - Support Vector Machine (SVM)
                % - Random Forest
                % - Neural Network
                % - Naive Bayes
                % - etc.
                W = train_classifier(X_train_inner, T_train_inner, num_train_inner, CLASSIFIER_METHOD);
                
                % Evaluate on validation fold
                Y_val = [X_val, ones(num_val, 1)] * W; % Discriminant values
                [~, predicted_class] = max(Y_val, [], 2); % Predicted class indices
                
                % Convert to one-hot encoding
                O_val = zeros(size(Y_val));
                O_val(sub2ind(size(Y_val), 1:num_val, predicted_class')) = 1;
                
                O_folds{fold} = O_val;
            end
            
            % Compute cross-validation accuracy for this feature combination
            T_val_all = cat(1, T_folds{:});
            O_val_all = cat(1, O_folds{:});
            confusion_matrix = T_val_all' * O_val_all;
            feature_accuracies(candidate_feature) = trace(confusion_matrix) / sum(confusion_matrix(:)) * 100;
            
        end % End candidate feature loop
        
        % Select best feature from this iteration
        [sorted_acc, sorted_idx] = sort(feature_accuracies, 'descend');
        
        % Find first feature not already selected
        for idx = 1:length(sorted_idx)
            if ~ismember(sorted_idx(idx), selected_features)
                selected_features = [selected_features, sorted_idx(idx)];
                inner_cv_accuracy(patient_idx, feature_count) = sorted_acc(idx);
                inner_cv_selected_features(patient_idx, feature_count) = sorted_idx(idx);
                break;
            end
        end
        
    end % End feature count loop
    
    % -------------------------------------------------------------------------
    % STEP 3: Determine optimal number of features (early stopping)
    % -------------------------------------------------------------------------
    % Stop at first local maximum to prevent overfitting
    optimal_feature_count(patient_idx) = find_optimal_feature_count(inner_cv_accuracy(patient_idx, :));
    optimal_features = inner_cv_selected_features(patient_idx, 1:optimal_feature_count(patient_idx));
    
    fprintf('  → Optimal feature count: %d (Accuracy: %.2f%%)\n', ...
            optimal_feature_count(patient_idx), ...
            inner_cv_accuracy(patient_idx, optimal_feature_count(patient_idx)));
    
    % -------------------------------------------------------------------------
    % STEP 4: Train final model on outer training set with optimal features
    % -------------------------------------------------------------------------
    X_train_optimal = X_train(:, optimal_features);
    X_test_optimal = X_test(:, optimal_features);
    
    % Compute class priors from combined train+test data
    K = sum(T_train_outer{patient_idx}, 1);
    priors = K / (num_train_samples + num_test_samples);
    
    % Train final classifier with optimal features
    W_final = train_lda_with_priors(X_train_optimal, T_train_outer{patient_idx}, ...
                                      num_train_samples, priors);
    
    % -------------------------------------------------------------------------
    % STEP 5: Evaluate on outer training set
    % -------------------------------------------------------------------------
    Y_train = [X_train_optimal, ones(num_train_samples, 1)] * W_final;
    [~, predicted_class] = max(Y_train, [], 2);
    O_train_outer{patient_idx} = zeros(size(Y_train));
    O_train_outer{patient_idx}(sub2ind(size(Y_train), 1:num_train_samples, predicted_class')) = 1;
    
    % -------------------------------------------------------------------------
    % STEP 6: Evaluate on outer test set (held-out patient)
    % -------------------------------------------------------------------------
    Y_test = [X_test_optimal, ones(num_test_samples, 1)] * W_final;
    [~, predicted_class] = max(Y_test, [], 2);
    O_test = zeros(size(Y_test));
    O_test(sub2ind(size(Y_test), 1:num_test_samples, predicted_class')) = 1;
    
    % Apply temporal smoothing (majority voting with window size 3)
    O_test_smoothed = apply_temporal_smoothing(O_test);
    O_test_outer{patient_idx} = O_test_smoothed;
    
    % Predict overall PSOC classification for this patient
    [max_count, predominant_class] = max(sum(O_test_smoothed));
    if predominant_class == 2 && max_count > PREDOMINANCE_THRESHOLD * size(O_test_smoothed, 1)
        PSOC_PREDICTED{patient_idx} = 'T';
    else
        PSOC_PREDICTED{patient_idx} = 'N';
    end
    
    fprintf('  → Predicted PSOC: %s (Manual: %s)\n\n', ...
            PSOC_PREDICTED{patient_idx}, PSOC_MANUAL{patient_idx});
    
end % End outer loop

fprintf('Nested cross-validation complete!\n\n');

%% ============================================================================
%% EVALUATE OVERALL PERFORMANCE
%% ============================================================================

% Aggregate confusion matrices across all patients
CM_train = cat(1, T_train_outer{:})' * cat(1, O_train_outer{:});
CM_test = cat(1, T_test_outer{:})' * cat(1, O_test_outer{:});

% Compute performance metrics
sensitivity_train = diag(CM_train) ./ sum(CM_train, 1)';
sensitivity_test = diag(CM_test) ./ sum(CM_test, 1)';
predictivity_train = diag(CM_train) ./ sum(CM_train, 2);
predictivity_test = diag(CM_test) ./ sum(CM_test, 2);
accuracy_train = trace(CM_train) / sum(CM_train(:)) * 100;
accuracy_test = trace(CM_test) / sum(CM_test(:)) * 100;

% PSOC classification agreement
psoc_agreement = sum(strcmp(PSOC_MANUAL, PSOC_PREDICTED));

%% Display Results
fprintf('===============================================\n');
fprintf('NESTED CROSS-VALIDATION RESULTS\n');
fprintf('===============================================\n');
fprintf('Training Set Performance:\n');
fprintf('  Sensitivity:   [%.2f%%, %.2f%%]\n', sensitivity_train * 100);
fprintf('  Predictivity:  [%.2f%%, %.2f%%]\n', predictivity_train * 100);
fprintf('  Accuracy:      %.2f%%\n\n', accuracy_train);

fprintf('Testing Set Performance (LOOCV):\n');
fprintf('  Sensitivity:   [%.2f%%, %.2f%%]\n', sensitivity_test * 100);
fprintf('  Predictivity:  [%.2f%%, %.2f%%]\n', predictivity_test * 100);
fprintf('  Accuracy:      %.2f%%\n\n', accuracy_test);

fprintf('PSOC Classification:\n');
fprintf('  Agreement: %d/%d (%.1f%%)\n', psoc_agreement, NUM_PATIENTS, psoc_agreement/NUM_PATIENTS*100);
fprintf('===============================================\n');

%% ============================================================================
%% HELPER FUNCTIONS
%% ============================================================================

function W = train_classifier(X, T, N, method)
    % Train a classifier on the given data
    % 
    % *** CLASSIFIER FLEXIBILITY ***
    % This function can be replaced with ANY classification algorithm:
    %   - Support Vector Machine (fitcsvm)
    %   - Random Forest (TreeBagger)
    %   - Neural Network (patternnet)
    %   - Naive Bayes (fitcnb)
    %   - k-Nearest Neighbors (fitcknn)
    %   - Logistic Regression (mnrfit)
    %   - Gradient Boosting (fitcensemble)
    %   - etc.
    %
    % Requirements:
    %   - Input: Feature matrix X (N x D), Target matrix T (N x C)
    %   - Output: Weight matrix W or model object
    %   - Must support prediction via: Y = [X, ones(N,1)] * W
    %
    % Inputs:
    %   X      - Feature matrix (N x D)
    %   T      - Target matrix (N x C), one-hot encoded
    %   N      - Number of samples
    %   method - 'LDA' or 'LR' (or add your own)
    %
    % Output:
    %   W      - Weight matrix (D+1 x C)
    
    if strcmp(method, 'LR')
        % Linear Regression (Least Squares)
        X_aug = [X, ones(N, 1)]; % Add bias term
        W = (X_aug' * X_aug) \ (X_aug' * T); % Normal equation
        
    elseif strcmp(method, 'LDA')
        % Linear Discriminant Analysis
        K = sum(T, 1); % Class counts
        priors = K / N; % Class priors
        
        % Class means: M = X^T * T * diag(1/K)
        means = X' * T * diag(1 ./ K);
        
        % Pooled covariance: CV = (X^T*X - M*diag(K)*M^T) / N
        covariance = (X' * X - means * diag(K) * means') / N;
        
        % Discriminant function: W = [CV\M; log(priors) - 0.5*diag(M^T*CV\M)]
        inv_cov_means = covariance \ means;
        W = [inv_cov_means; log(priors) - 0.5 * diag(means' * inv_cov_means)'];
        
    else
        error('Unknown classifier method: %s', method);
    end
end

function W = train_lda_with_priors(X, T, N, priors)
    % Train LDA with specified prior probabilities
    % (Used for final model training with domain-specific priors)
    %
    % Inputs:
    %   X      - Feature matrix (N x D)
    %   T      - Target matrix (N x C), one-hot encoded
    %   N      - Number of samples
    %   priors - Prior probabilities (1 x C)
    %
    % Output:
    %   W      - Weight matrix (D+1 x C)
    
    K = sum(T, 1);
    means = X' * T * diag(1 ./ K);
    covariance = (X' * X - means * diag(K) * means') / N;
    inv_cov_means = covariance \ means;
    W = [inv_cov_means; log(priors) - 0.5 * diag(means' * inv_cov_means)'];
end

function optimal_count = find_optimal_feature_count(accuracies)
    % Find the optimal number of features using early stopping criterion
    % Stops at first local maximum to prevent overfitting
    %
    % Strategy: Add features while accuracy increases, stop when it decreases
    %
    % Input:
    %   accuracies - Array of accuracies for increasing feature counts
    %
    % Output:
    %   optimal_count - Index of first local maximum
    
    for i = 1:(length(accuracies) - 1)
        if accuracies(i + 1) < accuracies(i)
            optimal_count = i;
            return;
        end
    end
    optimal_count = length(accuracies); % Use all features if monotonically increasing
end

function O_smoothed = apply_temporal_smoothing(O)
    % Apply temporal smoothing using majority voting with window size 3
    % Corrects isolated misclassifications in time-series data
    %
    % Rule: If frame i-1 and i+1 have same class, assign that class to frame i
    %
    % Input:
    %   O - Prediction matrix (N x C), one-hot encoded
    %
    % Output:
    %   O_smoothed - Smoothed prediction matrix (N x C)
    
    O_smoothed = O;
    
    % Convert one-hot to class indices for easier comparison
    [~, class_indices] = max(O, [], 2);
    
    % Apply majority voting: if neighbors agree, adopt their class
    for i = 1:(length(class_indices) - 2)
        if class_indices(i) == class_indices(i + 2) % Neighbors agree
            O_smoothed(i + 1, :) = O(i, :); % Adopt neighbor's class
        end
    end
end