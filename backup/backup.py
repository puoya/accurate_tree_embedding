# class FourLayerNN(nn.Module):
#     def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
#         super(FourLayerNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden1_dim)
#         self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
#         self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
#         self.fc4 = nn.Linear(hidden3_dim, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
######################################################################
######################################################################
######################################################################
# def train_neural_network(X, y, test_size=0.1, stratify=None, seed=None, num_iterations=20):
#     accuracies = []

#     np.random.seed(seed)  # For reproducibility of random splits
    
#     for iteration in range(num_iterations):
#         torch.manual_seed(seed+iteration)
        
#         state= np.random.randint(0, 10000)
#         #print(state)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=state)
    
#         X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#         y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#         X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

#         input_dim = X.shape[1]
#         hidden1_dim = 512
#         hidden2_dim = 128
#         hidden3_dim = 32
#         output_dim = len(np.unique(y))
#         model = FourLayerNN(input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim)

#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)

#         num_epochs = 100
#         for epoch in range(num_epochs):
#             model.train()
#             optimizer.zero_grad()
#             outputs = model(X_train_tensor)
#             loss = criterion(outputs, y_train_tensor)
#             loss.backward()
#             optimizer.step()

#         # Evaluate the model
#         model.eval()
#         with torch.no_grad():
#             outputs = model(X_test_tensor)
#             _, predicted = torch.max(outputs, 1)
#             accuracy = accuracy_score(y_test, predicted.numpy())
#             accuracies.append(accuracy)

#     average_accuracy = np.mean(accuracies)
#     return average_accuracy
# ######################################################################
# ######################################################################
# ######################################################################
# def train_svm_and_evaluate(X, y, seed):
#     np.random.seed(seed)  # For reproducibility of random splits
#     accuracies = []  # To store accuracy of each split
    
#     for _ in range(20):  # Perform 20 random splits and evaluations
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=np.random.randint(0, 10000))
#         model = SVC()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracies.append(accuracy_score(y_test, y_pred))
    
#     average_accuracy = np.mean(accuracies)
#     return average_accuracy
# ######################################################################
# ######################################################################
# ######################################################################
# def train_random_forest_and_evaluate(X, y, seed,max_tree_depth=20):
#     np.random.seed(seed)  # For reproducibility of random splits
#     accuracies = []  # To store accuracy of each split

#     max_d = 0
#     for _ in range(20):  # Perform 20 random splits and evaluations
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=np.random.randint(0, 10000))
#         #print(max_tree_depth)
#         model = RandomForestClassifier(max_depth=max_tree_depth)
#         model.fit(X_train, y_train)
#         #tree_depths = [tree.get_depth() for tree in model.estimators_]
#         #max_d = max( [max_d , max(tree_depths) ]) 
#         #print(max_d)
#         y_pred = model.predict(X_test)    
#         accuracies.append(accuracy_score(y_test, y_pred))
#     average_accuracy = np.mean(accuracies)
    
#     return average_accuracy
# ######################################################################
# ######################################################################
# ######################################################################
# def run_spherical_mds(dataset_name):
#     class parameters:
#         def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
#             self.D = D
#             self.d = d
#             self.N = N
#             self.sigma = sigma
#     ######################################################################
#     def load_data(dataset_name):
#         if dataset_name == 'GUniFrac':
#             address = "datasets/GUniFrac/doc/csv/"
#             X = np.load(address+'X.npy')
#             X = X.T
#         elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
#             address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
#             X = np.load(address+'X.npy')
#             X = X.T
#             N, D = np.shape(X)
#             for n in range(N):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#         elif dataset_name == 'document':
#             address = "datasets/document/"
#             X = np.load(address+'X.npy').astype(float)
#             address = "datasets/document/"
#             label = np.load(address+'label.npy')
#             idx = (label==0) + (label == 1)
#             X = X[idx,:]
#             label = label[idx]
#             np.random.seed(42)
#             N0 = 400
#             N = np.shape(X)[0]
#             idx = np.random.choice(N, N0, replace=False)
#             X = X[idx,:]
#             for n in range(N0):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#         return X
#     ######################################################################
#     directory = "datasets/" +dataset_name +"/results/" 
#     if not os.path.exists(directory):
#         os.makedirs(directory)
    
#     X = load_data(dataset_name)

#     # Initial parameters
#     D,N = np.shape(X)
#     D -= 1
    
#     DM = compute_sdm(X)
#     AIDM = compute_aidm(X)
#     KLDM = compute_kldm(X)
#     TVDM = compute_tvdm(X)
#     JSDM = compute_jsdm(X)

#     # A list of functions to run
#     functions = [
#         estimate_spherical_subspace,
#         estimate_spherical_subspace_liu,
#         estimate_spherical_subspace_dai,
#         #estimate_spherical_subspace_pga,
#         estimate_spherical_subspace_pga_2,
#     ]

#     # Prepare a list to collect the data
#     data_list = []
#     for d in range(2,D):
#         print(f"Running with d = {d}")
#         param = parameters(D=D, d=d, N=N, sigma = 0)
#         cnt = 0
#         for func in functions:
#             cnt = cnt +1
#             directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
#             if not os.path.exists(directory_):
#                 continue
#             X_ = np.load(directory_+'X_.npy')
#             S_ = np.load(directory_+'S_.npy', allow_pickle=True)

#             AIDM_ = compute_aidm(X_)
#             KLDM_ = compute_kldm(X_)
#             DM_ = compute_sdm(X_)
#             TVDM_ = compute_tvdm(X_)
#             JSDM_ = compute_jsdm(X_)

#             sphere_dist = total_distance(X, X_)
#             dist_distortion = distance_distortion(DM_, DM)
#             ai_distortion = distance_distortion(AIDM_, AIDM)
#             kl_distortion = distance_distortion(KLDM_, KLDM)
#             tv_distortion = distance_distortion(TVDM_, TVDM)
#             js_distortion = distance_distortion(JSDM_, JSDM)

#             data_list.append({"d": d, "Method": func.__name__, 
#                 "sphere_dist":sphere_dist,
#                 "ai_distortion":ai_distortion, 
#                 "kl_distortion":kl_distortion,
#                 "js_distortion":js_distortion,
#                 "tv_distortion":tv_distortion,
#                 "dist_distortion": dist_distortion
#                 })
#             print("[",sphere_dist, ai_distortion, kl_distortion, js_distortion, tv_distortion,dist_distortion, "]")
#             results_df = pd.DataFrame(data_list)
#         print('############################################')
#         #results_df.to_csv(results_filename, index=False)
#         if d >= min(np.shape(X))-1:
#             break

# ######################################################################
# ######################################################################
# ######################################################################
# def run_spherical_classifier(dataset_name):
#     class parameters:
#         def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
#             self.D = D
#             self.d = d
#             self.N = N
#             self.sigma = sigma
#     ######################################################################
#     def load_data(dataset_name):
#         if dataset_name == 'GUniFrac':
#             address = "datasets/GUniFrac/doc/csv/"
#             X = np.load(address+'X.npy')
#             X = X.T
#             y = np.load('datasets/GUniFrac/doc/csv/SmokingStatus_categories.npy')
#         elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
#             address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
#             X = np.load(address+'X.npy')
#             X = X.T
#             N, D = np.shape(X)
#             for n in range(N):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#             y = np.load('datasets/doi_10_5061_dryad_pk75d__v20150519/age_categories.npy') 
#         elif dataset_name == 'document':
#             address = "datasets/document/"
#             X = np.load(address+'X.npy').astype(float)
#             address = "datasets/document/"
#             label = np.load(address+'label.npy')
#             idx = (label==0) + (label == 1)
#             X = X[idx,:]
#             label = label[idx]
#             np.random.seed(42)
#             N0 = 400
#             N = np.shape(X)[0]
#             idx = np.random.choice(N, N0, replace=False)
#             X = X[idx,:]
#             y = label[idx]
#             for n in range(N0):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#         return X,y
#     ######################################################################
#     directory = "datasets/" +dataset_name +"/results/" 
#     #directory = "../results/" + dataset_name + "/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     results_filename = os.path.join(directory, "classification_results_rf.csv")
#     if os.path.exists(results_filename):
#         results_df = pd.read_csv(results_filename)
#         print("Results already exist. Loaded results:")
#     else:
#         print("Results file does not exist. Computing the results...")
#         X,y = load_data(dataset_name)
#         ind = (y != 3)
#         y = y[ind]
#         #y = y[ind]
#         #print(y)

#         D,N = np.shape(X)
#         D = D-1
        
#         functions = [
#             sfpca.estimate_spherical_subspace,
#             sfpca.estimate_spherical_subspace_liu,
#             sfpca.estimate_spherical_subspace_dai,
#             #sfpca.estimate_spherical_subspace_pga,
#             sfpca.estimate_spherical_subspace_pga_2,
#         ]

#         # Prepare a list to collect the data
#         data_list = []
#         seed = 100000
#         #average_acc_rf = train_random_forest_and_evaluate(X.T**2, y, seed=seed)

#         for d in range(2,D):
#             print(f"Running with d = {d}")
#             param = parameters(D=D, d=d, N=N, sigma = 0)
#             for func in functions:
#                 directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
#                 if not os.path.exists(directory_):
#                     continue
#                 X_ = np.load(directory_+'X_.npy')
#                 S_ = np.load(directory_+'S_.npy', allow_pickle=True)
#                 H = S_.item().H
#                 X_low = np.matmul(H.T,X_)

#                 X_ = X_[:,ind]
#                 X_low = X_low[:,ind]
                
#                 #average_acc_nn = train_neural_network(X_.T**2, y, seed=seed)
#                 #average_acc_svm = train_svm_and_evaluate(X_.T**2, y, seed=seed)
#                 average_acc_rf = train_random_forest_and_evaluate(X_.T**2, y, seed=seed)

#                 #average_acc_nn_low = train_neural_network(X_low.T**2, y, seed=seed)
#                 #average_acc_svm_low = train_svm_and_evaluate(X_low.T**2, y, seed=seed)
#                 #average_acc_rf_low = train_random_forest_and_evaluate(X_low.T**2, y, seed=seed)
                
#                 data_list.append({"d": d, "Method": func.__name__, 
#                     #"average_acc_nn":average_acc_nn,
#                     #"average_acc_svm":average_acc_svm,
#                     "average_acc_rf":average_acc_rf
#                     #"average_acc_nn_low":average_acc_nn_low,
#                     #"average_acc_svm_low":average_acc_svm_low,
#                     #"average_acc_rf_low":average_acc_rf_low
#                     })
#                 #print("[",average_acc_nn, average_acc_svm,average_acc_rf,average_acc_nn_low, average_acc_svm_low,average_acc_rf_low, "]")
#                 print("[",average_acc_rf, "]")
#                 results_df = pd.DataFrame(data_list)
#                 data_list.append({"d": d, "Method": func.__name__, 
#                     "average_acc_rf":average_acc_rf
#                     })
#                 #print("[",average_acc_rf, "]")
#                 results_df = pd.DataFrame(data_list)
#             print('############################################')
#             results_df.to_csv(results_filename, index=False)
#             if d >= min(np.shape(X))-1:
#                break
# ######################################################################
# def run_spherical_information(dataset_name):
#     class parameters:
#         def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
#             self.D = D
#             self.d = d
#             self.N = N
#             self.sigma = sigma
#     ######################################################################
#     def load_data(dataset_name):
#         if dataset_name == 'GUniFrac':
#             address = "datasets/GUniFrac/doc/csv/"
#             X = np.load(address+'X.npy')
#             X = X.T
#             y = np.load('datasets/GUniFrac/doc/csv/SmokingStatus_categories.npy')
#         elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
#             address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
#             X = np.load(address+'X.npy')
#             X = X.T
#             N, D = np.shape(X)
#             for n in range(N):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#             y = np.load('datasets/doi_10_5061_dryad_pk75d__v20150519/age_categories.npy') 
#         elif dataset_name == 'document':
#             address = "datasets/document/"
#             X = np.load(address+'X.npy').astype(float)
#             address = "datasets/document/"
#             label = np.load(address+'label.npy')
#             idx = (label==0) + (label == 1)
#             X = X[idx,:]
#             label = label[idx]
#             np.random.seed(42)
#             N0 = 400
#             N = np.shape(X)[0]
#             idx = np.random.choice(N, N0, replace=False)
#             X = X[idx,:]
#             y = label[idx]
#             for n in range(N0):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#         return X,y
#     ######################################################################
#     directory = "datasets/" +dataset_name +"/results/" 
#     #directory = "../results/" + dataset_name + "/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     results_filename = os.path.join(directory, "information_results.csv")
#     if os.path.exists(results_filename):
#         results_df = pd.read_csv(results_filename)
#         print("Results already exist. Loaded results:")
#     else:
#         print("Results file does not exist. Computing the results...")
#         X,y = load_data(dataset_name)
#         ind = (y != 3)
#         y = y[ind]

#         D,N = np.shape(X)
#         D = D-1
        
#         functions = [
#             sfpca.estimate_spherical_subspace,
#             sfpca.estimate_spherical_subspace_liu,
#             sfpca.estimate_spherical_subspace_dai,
#             #sfpca.estimate_spherical_subspace_pga,
#             sfpca.estimate_spherical_subspace_pga_2,
#         ]

#         # Prepare a list to collect the data
#         data_list = []
#         for d in range(2,D):
#             print(f"Running with d = {d}")
#             param = parameters(D=D, d=d, N=N, sigma = 0)
#             cnt = 0
#             for func in functions:
#                 cnt = cnt +1
#                 directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
#                 if not os.path.exists(directory_):
#                     continue
#                 X_ = np.load(directory_+'X_.npy')
#                 S_ = np.load(directory_+'S_.npy', allow_pickle=True)
#                 H = S_.item().H
#                 X_low = np.matmul(H.T,X_)

#                 X_ = X_[:,ind]
#                 X_low = X_low[:,ind]

#                 average_mi = compute_mutual_information(X_.T, y)
#                 average_mi_low = compute_mutual_information(X_low.T, y)

#                 data_list.append({"d": d, "Method": func.__name__, 
#                     "average_mi":average_mi,
#                     "average_mi_low":average_mi_low
#                     })
#                 print("[",average_mi, average_mi_low, "]")
#                 results_df = pd.DataFrame(data_list)
#             print('############################################')
#             results_df.to_csv(results_filename, index=False)
#             if d >= min(np.shape(X))-1:
#                 break


# def compute_sparsity(matrix, threshold=0):
#     if not isinstance(matrix, np.ndarray):
#         matrix = np.array(matrix)
#     zero_elements = np.sum(np.abs(matrix) <= threshold)
#     total_elements = matrix.size
#     sparsity_level = zero_elements / total_elements
    
#     return sparsity_level
# ######################################################################
# def run_spherical_sparsity(dataset_name):
#     class parameters:
#         def __init__(self, D = 1000, d = 1, N = 10, sigma = 0.01):
#             self.D = D
#             self.d = d
#             self.N = N
#             self.sigma = sigma
#     ######################################################################
#     def load_data(dataset_name):
#         if dataset_name == 'GUniFrac':
#             address = "datasets/GUniFrac/doc/csv/"
#             X = np.load(address+'X.npy')
#             X = X.T
#             y = np.load('datasets/GUniFrac/doc/csv/SmokingStatus_categories.npy')
#         elif dataset_name == 'doi_10_5061_dryad_pk75d__v20150519':
#             address = "datasets/doi_10_5061_dryad_pk75d__v20150519/"
#             X = np.load(address+'X.npy')
#             X = X.T
#             N, D = np.shape(X)
#             for n in range(N):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#             y = np.load('datasets/doi_10_5061_dryad_pk75d__v20150519/age_categories.npy') 
#         elif dataset_name == 'document':
#             address = "datasets/document/"
#             X = np.load(address+'X.npy').astype(float)
#             address = "datasets/document/"
#             label = np.load(address+'label.npy')
#             idx = (label==0) + (label == 1)
#             X = X[idx,:]
#             label = label[idx]
#             np.random.seed(42)
#             N0 = 400
#             N = np.shape(X)[0]
#             idx = np.random.choice(N, N0, replace=False)
#             X = X[idx,:]
#             y = label[idx]
#             for n in range(N0):
#                 X[n,:] = X[n,:]/ np.sum(X[n,:])
#                 X[n,:] = np.sqrt(X[n,:])
#             X = X.T
#         return X,y
#     ######################################################################
#     directory = "datasets/" +dataset_name +"/results/" 
#     results_filename = os.path.join(directory, "sparsity_results2.csv")
#     if os.path.exists(results_filename):
#         # Load the DataFrame from the CSV file
#         results_df = pd.read_csv(results_filename)
#         # Print the loaded results
#         print("Results already exist. Loaded results:")
#     else:
#         print("Results file does not exist. Computing the results...")
#         X,y = load_data(dataset_name)
#         # Initial parameters
#         D,N = np.shape(X)
#         D = D-1
#         # A list of functions to run
#         functions = [
#             sfpca.estimate_spherical_subspace,
#             sfpca.estimate_spherical_subspace_liu,
#             sfpca.estimate_spherical_subspace_dai,
#             #sfpca.estimate_spherical_subspace_pga,
#             sfpca.estimate_spherical_subspace_pga_2,
#         ]

#         # Prepare a list to collect the data
#         data_list = []
#         threshold = 10**(-4)/D
#         S = compute_sparsity(X**2, threshold)
#         for d in range(2,D):
#             #print(f"Running with d = {d}")
#             param = parameters(D=D, d=d, N=N, sigma = 0)
#             cnt = 0
#             for func in functions:
#                 cnt = cnt +1
#                 directory_ = directory+str(d)+'/'+str(func.__name__)+'/'
            
#                 if not os.path.exists(directory_):
#                     continue
#                 X_ = np.load(directory_+'X_.npy')
#                 #print(X_)
                
#                 S_ = compute_sparsity(X_**2-X**2, threshold)
                

#                 data_list.append({"d": d, "Method": func.__name__, 
#                     "sparsity_dist":np.abs(S-S_),
#                     "sparsity":S_
#                     })
#                 print("[",np.abs(S-S_), S_, "]")
#                 results_df = pd.DataFrame(data_list)
#             print('############################################')
#             results_df.to_csv(results_filename, index=False)
#             if d >= min(np.shape(X))-1:
#                 break

# cost = torch.norm(distance_matrix_paramterized - scale*distance_matrix,p='fro')**2/ torch.norm(scale*distance_matrix,p='fro')**2

    
    
    # errors = (torch.div(distance_matrix_paramterized , scale * distance_matrix) -1).abs()[mask]
    # errors_flat = errors.view(-1)
    # sorted_errors, _ = torch.sort(errors_flat)
    

    #print( int(epoch / 2000 * (n*(n-1)//2) ), n*(n-1)//2)
    # if epoch > threshold_epoch:
        # Flatten the errors and sort them
        # errors_flat = errors.view(-1)
        # sorted_errors, _ = torch.sort(errors_flat)

        # distances_flat = distance_matrix[mask].view(-1)
        # sorted_distances, _ = torch.sort(-distances_flat)
        # sorted_distances = -sorted_distances


        
        # Determine the number of elements to discard
        # discard_count = int( (epoch-threshold_epoch) / 2000 * (n*(n-1)//2) )
        
        # Determine the threshold value for the discard
        # if discard_count > 0:
        #     # threshold_value = sorted_errors[discard_count]
        #     threshold_value = sorted_distances[discard_count]

        #     # Create a mask to discard smallest errors
        #     # discard_mask = errors <= threshold_value
        #     keep_mask = distance_matrix <= threshold_value
        # else:
        #     # discard_mask = torch.ones_like(errors, dtype=torch.bool)
        #     keep_mask = mask
        # print(keep_mask)
            
        # cost = torch.norm(distance_matrix_paramterized[keep_mask] - scale*distance_matrix[keep_mask],p='fro')**2/torch.norm(scale*distance_matrix[keep_mask],p='fro')**2

        # a = (1/distance_matrix-1)
        # b = 1
        # W = a * ((epoch-threshold_epoch)/(2000-threshold_epoch))**5 + b
        # W = 1/(scale*distance_matrix)
        # W.fill_diagonal_(1)
        # print(W)
        # cost = torch.norm( torch.mul(distance_matrix_paramterized - scale*distance_matrix,W),p='fro')**2/torch.norm(torch.mul(scale*distance_matrix,W),p='fro')**2
        # W = generate_weighted_mask(distance_matrix, 1-(epoch-threshold_epoch)/(2000-threshold_epoch))
        # # print(W)
        # cost = torch.norm( torch.mul(distance_matrix_paramterized - scale*distance_matrix,W),p='fro')**2

        # Apply the mask to the errors
        #cost = torch.sum(torch.pow(errors[discard_mask],2))
    
    # cost = torch.norm(errors, p='fro') ** 2 / torch.norm(scale * distance_matrix, p='fro') ** 2
    

    #cost = torch.mean( torch.pow(torch.div(distance_matrix_paramterized,scale*distance_matrix )[mask]-1,2) )
    
    # difference_matrix = torch.pow(distance_matrix_paramterized - scale*distance_matrix,2)
    # scale_distance_matrix_2 = torch.pow(scale*distance_matrix,2)
    



    #cost = torch.sum( torch.pow(D_X - grammian_matrix,2) )/ torch.sum(torch.pow(grammian_matrix,2) )
    #print(cost)

    #cost = torch.sum(difference_matrix[mask])/torch.sum(scale_distance_matrix_2[mask])
    

    # elif feed_scale is None:
        # scale = torch.div( torch.sum(torch.mul(distance_matrix_paramterized,distance_matrix_paramterized)),
            # torch.sum(torch.mul(distance_matrix_paramterized,distance_matrix)))
        # pdist = torch.div(distance_matrix_paramterized, distance_matrix)[W]
        # scale = torch.div(torch.sum(torch.pow(pdist,2)) , torch.sum(pdist) )
        

        # if weights is not None:
        #     weight = weights[i]
        # else:
        #     weight = 1
        # if i < cnt:
        #     weight *= 1
        # else:
        #     if np.abs(i-cnt) < window:
        #         weight *= p**np.abs(cnt-i)
        #     else:
        #         p = 0
        #         weight *= p**np.abs(cnt-i)
        # if i <= cnt:
        #     weight = 1
        # else:
        #     weight = 1/np.sqrt(np.abs(i-cnt+1))
        # weight = 1
        # if cnt < window:
        #     if np.abs(cnt-i) < window:
        #         weight = p**(np.abs(cnt-i))
        #     else:
        #         weight = 10**(-20) # drastically small
        # else:
        #     small_window = (0.01)*len(masks) # value = 1/200 at 2 (out of 200) 
        #     ratio = np.sqrt( (cnt-window)/(len(masks)-1-window) ) # current progress ... make the window smaller
            
        #     front_window =  ratio * small_window + (1-ratio) * window
        #     pf = np.exp(np.log(threshold)/ (front_window) )

        #     if i < cnt:
        #         if np.abs(cnt-i) < window:
        #             weight = p**(np.abs(cnt-i))
        #         else:
        #             weight = threshold
        #     else:
        #         weight = pf**(np.abs(cnt-i))
        # weight = 1
        #cost += weight*torch.log10(cost_i)
        # new_weights.append(weight)
        # weights.append(weight)
    
    # cost /= np.sum(new_weights)
    # else:

        #     # _, _,_,costs = cost_function(tangents, distance_matrix, masks, scale_update_mode, cnt )
            
        #     training_epochs = int(0.05*epochs)
        #     initial_target = 0
        #     final_target = num_bins-1
            
        #     initial_cnt = epochs_to_estimate_curvature + training_epochs
        #     final_cnt = epochs
        #     if epoch < initial_cnt:
        #         cnt = 0
        #     elif epoch < final_cnt:
        #         a = (final_target-initial_target)/(final_cnt-initial_cnt)
        #         b = initial_target - a* initial_cnt
        #         cnt = int(a * epoch + b)
        #     print(f'cnt is {cnt}')
        #     for param_group in optimizer_tangents.param_groups:
        #         # r = (epoch/epochs)
        #         param_group['lr'] = lrs[cnt]*0.03 #r*decrease_factor2+(1-r)*decrease_factor1



        #     # Optimize tangents
        #     optimizer_tangents.zero_grad()
        #     # loss, scale , distance_matrix_paramterized,costs = cost_function(tangents, distance_matrix, masks, scale_update_mode,cnt,feed_scale = torch.tensor(scale.item()))
        #     lr = np.log(lrs[cnt])
        #     min_lr = np.log(np.min(lrs))
        #     max_lr = np.log(np.max(lrs))
        #     rate = 1-(lr - min_lr)/(max_lr - min_lr)
        #     print(f'rate is {rate}')
        #     # loss, scale , distance_matrix_paramterized, costs = cost_function(tangents, distance_matrix, masks, scale_update_mode,cnt, r = rate)


        #     loss.backward()

        #     with torch.no_grad():
        #         tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)

        #     optimizer_tangents.step()

            
        #     # bin_means = compute_binned_costs(distance_matrix_paramterized, scale, distance_matrix, num_bins)
        #     # Clear previous plot and update with new data
        #     ax1.clear()
        #     ax2.clear()

        #     # Plot costs on the first subplot
        #     colors = ['#6495ED'] * num_bins  # Use medium blue for default color
        #     colors[cnt] = '#FF0000'  # Use vibrant red for the bin at index `cnt`
        #     ax1.bar(range(num_bins), costs, color=colors)
        #     ax1.set_ylabel('Costs', color='blue')
        #     ax1.set_ylim(-5, 6)  # Set consistent y-axis limits
        #     ax1.tick_params(axis='y', labelcolor='blue')

        #     # Plot means on the second subplot
        #     ax2.bar(range(num_bins), np.log10(means), color='green', alpha=0.5)
        #     ax2.set_xlabel('Bin')
        #     ax2.set_ylabel('Means', color='green')
        #     ax2.set_ylim(-6, 1)  # Set consistent y-axis limits
        #     ax2.tick_params(axis='y', labelcolor='green')

        #     # Title and adjust layout
        #     fig.suptitle(f'Epoch {epoch + 1}')
        #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        #     # Save figure
        #     save_dir = "training_frames"
        #     fig_path = os.path.join(save_dir, f"epoch_{epoch + 1:04d}.png")
        #     fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)  # Use tight layout and minimal padding

        #     # Update the figure display
        #     clear_output(wait=True)
        #     display(fig)
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

        #     # Optionally, pause for a short time to allow display to update
        #     plt.pause(0.01)



        #     # custom_scheduler.step(loss)
        #     current_lr = optimizer_tangents.param_groups[0]['lr']
        #     print(f"Epoch {epoch}, Loss: {loss.item()}, Scale: {scale.item()}, Learning Rate: {current_lr}")




            # else:
            #     loss, scale , distance_matrix_paramterized, costs = cost_function(tangents, distance_matrix, masks, scale_update_mode,cnt, weights = weights)
            # weights = []
            # for i in range(len(costs)):
            #     weights.append(10**(-costs[i]))
            
            # bin_means = compute_binned_costs(distance_matrix_paramterized, scale, distance_matrix, num_bins)
            # clear_output(wait=True)
            # ax.clear()
            # ax.bar(range(num_bins), bin_means)
            # ax.set_title(f'Epoch {epoch + 1}')
            # ax.set_xlabel('Bin')
            # ax.set_ylabel('Mean Cost')
            # display(fig)
            # plt.pause(0.01)

            # custom_scheduler.step(loss)
            # current_lr = optimizer_tangents.param_groups[0]['lr']
            # print(f"Epoch {epoch}, Loss: {loss.item()}, Scale: {scale.item()}, Learning Rate: {current_lr}")


            # size = distance_matrix.size()
    # W = torch.ones(size, dtype=torch.bool)
    # # Set the diagonal elements to False
    # W.fill_diagonal_(False)

    # if r is not None:
    #     matrix = torch.pow(scale*distance_matrix , -r)
    # else:
    #     matrix = torch.ones_like(distance_matrix)

    # print(epoch,torch.max(matrix[W]).item(), torch.min(matrix[W]).item())
    # cost = torch.norm( torch.mul(distance_matrix_paramterized[W] - scale*distance_matrix[W], matrix[W]),p='fro')**2 /torch.norm( matrix[W],p='fro')**2


    # window = (0.35)*len(masks) # value = 1/200 at 50 (out of 200) exponential model
    # threshold = 1/len(masks)    
    # p = np.exp(np.log(threshold)/window)
    # new_weights = []

    # means = []
    # lrs = []
    # for i in range(len(masks)):
        # W = masks[i]
        # means.append(torch.mean(distance_matrix[W]).item())
        # if i == 0:
        #     lrs.append(1)
        # else:
        #     lrs.append(means[i]/means[0])
    












# from scipy.stats import wasserstein_distance
# import geom.poincare as poincare
# import geom.euclidean as euclidean
# from learning.frechet import Frechet
# from scipy.sparse.linalg import eigs
# import geom.hyperboloid as hyperboloid
# from geom.horo import busemann, project_kd
# from learning.pca import EucPCA, TangentPCA,PGA, HoroPCA, BSA
# from utils.metrics import compute_metrics


# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score
# # import spaceform_pca_lib as sfpca


# from sklearn.feature_selection import mutual_info_classif
# from sklearn.preprocessing import OneHotEncoder
# from scipy.stats import pearsonr


# import torch.nn as nn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier

# import matplotlib.pyplot as plt
# from IPython.display import display, clear_output





# def compute_binned_costs(distance_matrix_paramterized, scale, distance_matrix, num_bins = 10):
#     # Compute element-wise operation
#     element_wise_cost = (distance_matrix_paramterized - scale * distance_matrix) ** 2 / (scale * distance_matrix) ** 2

#     # Extract upper triangular elements
#     triu_indices = torch.triu_indices(distance_matrix.size(0), distance_matrix.size(1), offset=1)
#     upper_triangular_values = element_wise_cost[triu_indices[0], triu_indices[1]]

#     # Sort the values
#     sorted_values, _ = torch.sort(upper_triangular_values)

#     # Compute bin size
#     bin_size = sorted_values.size(0) // num_bins

#     # Initialize list to store mean values of bins
#     bin_means = []

#     for i in range(num_bins):
#         if i == num_bins - 1:
#             # Last bin takes the remaining elements
#             bin_values = sorted_values[i * bin_size:]
#         else:
#             bin_values = sorted_values[i * bin_size:(i + 1) * bin_size]
#         bin_means.append( torch.log(bin_values.mean()).item())

#     return bin_means
###########################################################################
###########################################################################
###########################################################################
# def generate_weighted_mask(distance_matrix,portion):
#     # Compute the probability matrix (weights are inverse of distance)
#     prob_matrix = 1 / (distance_matrix)  # Add small value to avoid division by zero
#     prob_matrix.fill_diagonal_(0)
    
#     prob_matrix /= prob_matrix.sum()  # Normalize to make it a probability distribution

#     # Flatten the probability matrix
#     prob_flat = prob_matrix.flatten().cpu().numpy()
#     nonzero_prob_flat = np.sum(prob_flat>0)


#     # Number of elements to select
#     num_elements = prob_flat.size
#     num_to_select = int(num_elements * portion)
#     # print(nonzero_prob_flat,num_to_select,num_elements)
#     if num_to_select > nonzero_prob_flat:
#         num_to_select = nonzero_prob_flat

#     # Generate random indices based on the probability distribution
#     selected_indices = np.random.choice(num_elements, num_to_select, replace=False, p=prob_flat)

#     # Initialize mask
#     mask = torch.zeros(num_elements, dtype=torch.bool, device=distance_matrix.device)
#     mask[selected_indices] = True
    
#     # Reshape mask to original distance matrix shape
#     mask = mask.view_as(distance_matrix)
#     prob_matrix[~mask] = 0

#     prob_matrix = (prob_matrix + prob_matrix.t())/2
#     prob_matrix /= prob_matrix.sum()

#     return prob_matrix
# ###########################################################################
# ###########################################################################
# ###########################################################################
# def create_bin_masks(distance_matrix, num_bins):
#     # Get the number of elements
#     n = distance_matrix.size(0)

#     # Flatten the upper triangular part (excluding the diagonal)
#     triu_indices = torch.triu_indices(n, n, offset=1)
#     upper_triangular_values = distance_matrix[triu_indices[0], triu_indices[1]]
    
#     # Sort the upper triangular values and get the sorted indices
#     sorted_values, sorted_indices = torch.sort(upper_triangular_values,descending= True)

    
#     # Compute the size of each bin
#     bin_size = len(sorted_values) // num_bins
#     remainder = len(sorted_values) % num_bins
    
#     # Initialize list to hold mask matrices
#     masks = []
    
#     start_idx = 0
#     for i in range(num_bins):
#         end_idx = start_idx + bin_size + (1 if i < remainder else 0)
        
#         # Create a mask for the current bin
#         mask = torch.zeros(n, n, dtype=torch.bool)
#         mask_values = sorted_indices[start_idx:end_idx]
#         mask[triu_indices[0][mask_values], triu_indices[1][mask_values]] = True
#         mask = mask | mask.t()  # Make the mask symmetric
        
#         masks.append(mask)
        
#         start_idx = end_idx
    
#     return masks
# ###########################################################################
# ###########################################################################
# ###########################################################################
# def create_clustered_masks(distance_matrix, num_clusters):
#     # Get the number of elements
#     n = distance_matrix.size(0)

#     # Flatten the upper triangular part (excluding the diagonal)
#     triu_indices = torch.triu_indices(n, n, offset=1)
#     upper_triangular_values = distance_matrix[triu_indices[0], triu_indices[1]].cpu().numpy()

#     # Reshape the data for clustering
#     upper_triangular_values = upper_triangular_values.reshape(-1, 1)

#     # Perform K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#     cluster_labels = kmeans.fit_predict( np.log10(upper_triangular_values) )
#     centroids = kmeans.cluster_centers_.flatten()
#     print(centroids)

#     # Sort clusters by centroid values in descending order
#     sorted_clusters = sorted(range(num_clusters), key=lambda x: centroids[x], reverse=True)
#     print(sorted_clusters)


#     # Initialize list to hold mask matrices
#     masks = []

#     for cluster in sorted_clusters:
#         mask = torch.zeros(n, n, dtype=torch.bool)
#         mask_indices = (cluster_labels == cluster)
#         mask_values = torch.tensor(mask_indices, dtype=torch.bool)

#         mask[triu_indices[0][mask_values], triu_indices[1][mask_values]] = True
#         mask = mask | mask.t()  # Make the mask symmetric

#         masks.append(mask)

#     return masks
# ###########################################################################
# ###########################################################################
# ###########################################################################

# def cost_function2(tangents, distance_matrix, clusteres_mask = None, scale_learning= False, scale = torch.tensor(1), unweighted = True, W = None):
#     n = tangents.size(1)

#     grammian_matrix = -torch.cosh(distance_matrix)
#     embedding = hyperbolic_exponential_torch(tangents)

#     embedding_ = embedding.clone()
#     embedding_[0, :] *= -1
#     G = torch.matmul(embedding.t(), embedding_)
#     G_clipped = torch.clamp(G, max=-1)
#     distance_matrix_paramterized = torch.arccosh(-G_clipped)

#     size = distance_matrix.size()
#     off_diagonals = torch.ones(size, dtype=torch.bool)
#     off_diagonals.fill_diagonal_(False)

#     cost = 0
#     costs = []

#     if scale_learning is not False:
#         scale = torch.div( torch.sum(torch.pow(distance_matrix_paramterized,2)),torch.sum(torch.mul(distance_matrix_paramterized,distance_matrix)))

#     if unweighted is True:
#         cost = (torch.norm( distance_matrix_paramterized[off_diagonals] - scale*distance_matrix[off_diagonals],p='fro')**2 )/ (torch.norm(scale*distance_matrix[off_diagonals],p='fro')**2) 
#     else:
#         if W is not None:
#             distance_matrix_paramterized_w = torch.mul(distance_matrix_paramterized,W)
#             distance_matrix_w = torch.mul(distance_matrix,W)
#             cost = (torch.norm( distance_matrix_paramterized_w[off_diagonals] - scale*distance_matrix_w[off_diagonals],p='fro')**2 )/ (torch.norm(scale*distance_matrix_w[off_diagonals],p='fro')**2) 
#         else:
#             print('ERROR')
#             return None, None, None
    
#     for i in range(len(clusteres_mask)):
#         mask_i = clusteres_mask[i]
#         cost_i = (torch.norm( distance_matrix_paramterized[mask_i] - scale*distance_matrix[mask_i],p='fro')**2 )/ (torch.norm(scale*distance_matrix[mask_i],p='fro')**2)
#         costs.append(torch.log10(cost_i).item())
        
        
#     return cost, scale, costs
###########################################################################
###########################################################################
###########################################################################
# def precise_hyperbolic_mds(dataset_name, dimension=None):
#     # Log and print the start of the Naive Hyperbolic Embedding Step
#     logging.info("Precise Hyperbolic Embedding Step.")
#     tqdm.write("Precise Hyperbolic Embedding Step.")

#     # Log and print the name of the dataset being processed
#     tqdm.write(f"Dataset: {dataset_name}")
#     logging.info(f"Dataset: {dataset_name}")

#     # Define the input directory for distance matrices
#     input_directory = f'datasets/{dataset_name}/distance_matrices'
#     # Check if the input directory exists
#     if not os.path.exists(input_directory):
#         tqdm.write("The input directory does not exist.")
#         logging.info("The input directory does not exist.")
#         return

#     # Define the output directory for hyperbolic points
#     output_directory = f'datasets/{dataset_name}/exact_hyperbolic_points'
#     # Create the output directory if it does not exist
#     os.makedirs(output_directory, exist_ok=True)

#     # Get all .npy files in the input directory
#     npy_files = [f for f in os.listdir(input_directory) if f.endswith(".npy")]

#     # If no .npy files are found, log and exit
#     if len(npy_files) == 0:
#         tqdm.write("No .npy files found in the directory.")
#         logging.info("No .npy files found in the directory.")
#         return

#     # Set up a progress bar for processing files
#     cnt = 0
#     with tqdm(total=len(npy_files), unit="file", dynamic_ncols=True) as pbar:
#         for filename in npy_files:
#             file_path = os.path.join(input_directory, filename)
#             # Load the distance matrix from the .npy file
#             distance_matrix = np.load(file_path)
#             distance_matrix = torch.tensor(distance_matrix)
#             scale = 10 / torch.max(distance_matrix)
#             matrix_scaled = scale * distance_matrix

#             grammian_matrix = -torch.cosh(matrix_scaled)
            
#             initial_embedding = np.load(f'datasets/{dataset_name}/hyperbolic_points/X_{filename[7:-4]}.npy')
#             initial_embedding = torch.tensor(initial_embedding)
#             initial_tangents = hyperbolic_log_torch(initial_embedding)
            
            save_dir = "training_frames"
            video_filename = "training_video.mp4"

            # Create a list of image file paths
            epochs = 2000
            image_files = [os.path.join(save_dir, f"epoch_{epoch + 1:04d}.png") for epoch in range(epochs)]
            
            p = 1/2

            total_time = 1
            x = 0
            # for epoch in range(epochs):
            #     x += int((epochs + 1)**p / (epoch + 1)**p)
            for epoch in range(epochs):
                x += int(1)

            x /= (total_time*60)
            x = int(x)


            # Create a video writer object
            with imageio.get_writer(video_filename, fps=x) as writer:
                for epoch, image_file in enumerate(image_files):
                    if epoch == 0:
                        continue
                    image = imageio.imread(image_file)
                    # Calculate frame count based on 1/sqrt(epoch)
                    frame_count = int((epochs + 1)**p / (epoch + 1)**p)
                    for _ in range(frame_count):
                        writer.append_data(image)
            print(f"Video saved as {video_filename}")

#             # matrix_scaled,initial_tangents=initial_tangents, epochs=2000, dim = dimension
#             # if cnt == 0:
#             tangents, scale = optimize_embedding(matrix_scaled,initial_tangents=initial_tangents, epochs=2000, dim = dimension)
#             embedding = hyperbolic_exponential_torch(tangents)
#             if scale.requires_grad:
#                 s = scale.detach().numpy()
#             else:
#                 s = scale.numpy()
#             np.save(f'{output_directory}/X_{filename[7:-4]}.npy', embedding.detach().numpy())
#             np.save(f'{output_directory}/scale_{filename[7:-4]}.npy', s)
            
            

#             # np.save(output_X_file_path, X)
            
#             # Log the processed file
#             logging.info(f"Processed file: {filename}")
#             pbar.set_postfix({"file": filename})
#             pbar.update(1)
###########################################################################
###########################################################################
###########################################################################








# Example usage
# if __name__ == "__main__":
    
    
    
#     # T = multitrees.contents[0].copy()
#     # processor = TreeProcessor(T)
#     # print(processor.contents)
#     # print(processor.embed_hyperbolic(dimension = 10))
#     embedd = Embedding(geometry = 'hyperbolic',model ='loid')
#     X = np.random.randn(3, 10)/100

#     # torch_tensor = torch.from_numpy(X)


#     # embedd.points = torch_tensor
#     embedd.points = X
#     b = np.random.randn(3)/10
    
#     print(embedd.points)
#     embedd.translate(b)
#     print(embedd.points)
#     embedd.translate(-b)
#     print(embedd.points)
    
    # X = processor.embed_hyperbolic(dimension = 200)
    # print(processor.tree.distance_matrix())

    # def apply_mds_embedding(self, n_components=2):
    #     self.log_info(f"Applying MDS embedding with {n_components} components...")
    #     # Apply multidimensional scaling (MDS) for embedding into Euclidean space
    #     distance_matrix = self.get_distance_matrix()
    #     mds = MDS(n_components=n_components, dissimilarity='precomputed')
    #     embedded_coords = mds.fit_transform(distance_matrix)

    #     return embedded_coords

    # def evaluate_embedding_quality(self, embedded_coords):
    #     self.log_info("Evaluating embedding quality...")
    #     # Evaluate the quality of embedding using trustworthiness
    #     distance_matrix_original = squareform(pdist(self.original_tree.distance_matrix()))
    #     distance_matrix_embedded = pairwise_distances(embedded_coords, metric='euclidean')
        
    #     return trustworthiness(distance_matrix_original, distance_matrix_embedded)







#             # save_dir = "training_frames"
#             # video_filename = "training_video.mp4"

#             # # Create a list of image file paths
#             # epochs = 2000
#             # image_files = [os.path.join(save_dir, f"epoch_{epoch + 1:04d}.png") for epoch in range(epochs)]
            
#             # p = 1/2

#             # total_time = 1
#             # x = 0
#             # # for epoch in range(epochs):
#             # #     x += int((epochs + 1)**p / (epoch + 1)**p)
#             # for epoch in range(epochs):
#             #     x += int(1)

#             # x /= (total_time*60)
#             # x = int(x)


#             # # Create a video writer object
#             # with imageio.get_writer(video_filename, fps=x) as writer:
#             #     for epoch, image_file in enumerate(image_files):
#             #         if epoch == 0:
#             #             continue
#             #         image = imageio.imread(image_file)
#             #         # Calculate frame count based on 1/sqrt(epoch)
#             #         frame_count = int((epochs + 1)**p / (epoch + 1)**p)
#             #         for _ in range(frame_count):
#             #             writer.append_data(image)
#             # print(f"Video saved as {video_filename}")

#             # matrix_scaled,initial_tangents=initial_tangents, epochs=2000, dim = dimension
#             # if cnt == 0:














# print(epoch,learning_rate(epoch,scale_range, init_lr))

# if epoch % 100 == 99:
            #     loss, scale, costs = cost_function2(tangents, distance_matrix, clusteres_mask = masks, scale_learning = True, unweighted = True)
            # else:

# if epoch % 100 == 99:
            #     loss, _, costs = cost_function2(tangents, distance_matrix, clusteres_mask = masks, scale_learning = False, scale = scale_grad_free, unweighted = False, W= W)
            # else:

# masks = create_bin_masks(distance_matrix, num_bins)
    # masks = create_clustered_masks(distance_matrix, num_bins)

# means = []
    # for i in range(len(masks)):
    #     W = masks[i]
    #     means.append(torch.mean(distance_matrix[W]).item())
    

    # Initialize the figure and axes outside the loop
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # fig.set_size_inches(12, 8)  # Increase figure size as needed (in inches)
    # fig.set_dpi(600)  # Increase DPI for higher resolution

    # # Reduce the space between subplots
    # plt.subplots_adjust(hspace=0.1)  # Adjust hspace as needed
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the figure
    # display(fig)

    # save_dir = "training_frames"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

# if epoch % 100 == 99:
        #     ax1.clear()
        #     ax2.clear()

        #     # Plot costs on the first subplot
        #     colors = ['#6495ED'] * num_bins  # Use medium blue for default color
        #     # colors[cnt] = '#FF0000'  # Use vibrant red for the bin at index `cnt`
        #     ax1.bar(range(num_bins), costs, color=colors)
        #     ax1.set_ylabel('Costs', color='blue')
        #     ax1.set_ylim(-5, 6)  # Set consistent y-axis limits
        #     ax1.tick_params(axis='y', labelcolor='blue')

        #     # Plot means on the second subplot
        #     ax2.bar(range(num_bins), np.log10(means), color='green', alpha=0.5)
        #     ax2.set_xlabel('Bin')
        #     ax2.set_ylabel('Means', color='green')
        #     ax2.set_ylim(-6, 1)  # Set consistent y-axis limits
        #     ax2.tick_params(axis='y', labelcolor='green')

        #     # Title and adjust layout
        #     fig.suptitle(f'Epoch {epoch + 1}')
        #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        #     # Save figure
        #     save_dir = "training_frames"
        #     fig_path = os.path.join(save_dir, f"epoch_{epoch + 1:04d}.png")
        #     fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.1)  # Use tight layout and minimal padding

        #     # Update the figure display
        #     clear_output(wait=True)
        #     display(fig)
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

        #     # Optionally, pause for a short time to allow display to update
        #     plt.pause(0.01)

# ###########################################################################
# ###########################################################################
# #optimizer = optim.Adam([tangents], lr=learning_rate)
# def update_scale(tangents, distance_matrix,scale):
#     n = tangents.size(1)
#     scaled_distance_matrix = scale*distance_matrix
#     embedding = hyperbolic_exponential_torch(tangents)

#     embedding_1 = embedding.clone()
#     embedding_1[0, :] *= -1
#     XTX = torch.matmul(embedding.t(), embedding_1)
#     XTX_clipped = torch.clamp(XTX, max=-1)
#     distance_matrix_paramterized = torch.arccosh(-XTX_clipped)

#     scale = torch.div( torch.sum(torch.mul(distance_matrix_paramterized,scaled_distance_matrix)), 
#             torch.sum(torch.mul(scaled_distance_matrix,scaled_distance_matrix)))

#     return scale
# ###########################################################################
# ###########################################################################
# ###########################################################################
#     #print('scaled',distance_matrix_paramterized)
#     #print(scale)

#     #cost = torch.norm(XTX_clipped - grammian_matrix) / torch.norm(grammian_matrix)
#     #cost = torch.sum( torch.pow(D_X - grammian_matrix,2) )/ torch.sum(torch.pow(grammian_matrix,2) )
#     #print(cost)
    

# def compute_pca_results(dataset_name, method):
#     """
#     Compute PCA results for a given dataset using the specified method.

#     Parameters:
#     dataset_name (str): Name of the dataset.
#     method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

#     Returns:
#     None
#     """
#     logging.info("Hyperbolic PCA Step.")
#     tqdm.write("Hyperbolic PCA Step.")

#     # Check if the provided method is valid
#     if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
#         tqdm.write(f"Method {method} is not supported.")
#         logging.error(f"Method {method} is not supported.")
#         return
    
#     # Define directory paths
#     distance_file = 'hyperbolic_points'
#     input_directory = f'datasets/{dataset_name}/{distance_file}' 

#     if not os.path.exists(input_directory):
#         tqdm.write(f"The input directory does not exist.")
#         logging.info(f"The input directory does not exist.")
#         return

#     output_directory = f'datasets/{dataset_name}/{method}/subspaces' 
    
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)

#     logging.info(f"Dataset: {dataset_name} \t Method: {method}")
#     tqdm.write(f"Dataset: {dataset_name} \t Method: {method}")

#     # List all .npy files in the input directory that start with 'X'
#     npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
#     if len(npy_files) == 0:
#         tqdm.write(f"No .npy files found in the directory.")
#         logging.info(f"No .npy files found in the directory.")
#         return

#     # Set up a progress bar to show the progress of processing files
#     with tqdm(total=len(npy_files), unit="file", dynamic_ncols=True) as pbar:    
#         for filename in npy_files:
#             try:
#                 # Get the full path of the current file
#                 file_path = os.path.join(input_directory, filename)
#                 # Extract the index from the filename (strip 'X' and '.npy')
#                 file_index = filename[2:-4]
#                 # Load the data from the .npy file
#                 X = np.load(file_path)
                
#                 # Perform different processing based on the method
#                 if method in ['sfpca', 'pga']:
#                     if method == 'sfpca':
#                         # Estimate the hyperbolic subspace using 'sfpca'
#                         S = estimate_hyperbolic_subspace(X)
#                     else:
#                         # Estimate the hyperbolic subspace using 'pga'
#                         S = estimate_hyperbolic_subspace_pga(X)
#                     # Save the estimated subspace object as a .pkl file
#                     with open(os.path.join(output_directory, 'subspace_' + file_index + '.pkl'), 'wb') as file:
#                         pickle.dump(S, file)
#                 elif method in ['horopca', 'bsa']:
#                     # Run dimensionality reduction using the specified method
#                     Q, mu = run_dimensionality_reduction(method, X)
#                     # Save the results as .pt files
#                     torch.save(Q, os.path.join(output_directory, 'Q_' + file_index + '.pt'))
#                     torch.save(mu, os.path.join(output_directory, 'mu_' + file_index + '.pt'))

#                 # Update the progress bar with the current file being processed
#                 pbar.set_postfix({"file": filename})
#                 pbar.update(1)
#                 logging.info(f"Processed file: {filename}")

#             except Exception as e:
#                 # Log any errors that occur during processing
#                 logging.error(f"Error processing file {filename}: {e}")
#                 continue
# ###########################################################################
# ###########################################################################
# ###########################################################################
# def compute_mds_results(dataset_name, method):
#     """
#     Compute MDS results for a given dataset using the specified method.

#     Parameters:
#     dataset_name (str): Name of the dataset.
#     method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

#     Returns:
#     None
#     """
#     logging.info("Compute MDS Costs.")
#     tqdm.write("Compute MDS Costs.")

#     # Check if the provided method is valid
#     if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
#         tqdm.write(f"Method {method} is not supported.")
#         logging.error(f"Method {method} is not supported.")
#         return

#     # Define directory paths
#     distance_file = 'hyperbolic_points'
#     base_directory = f'datasets/{dataset_name}'
#     subspace_directory = f'{base_directory}/{method}/subspaces'
#     input_directory = f'{base_directory}/{distance_file}'

#     if not os.path.exists(input_directory):
#         tqdm.write(f"The input directory does not exist.")
#         logging.info(f"The input directory does not exist.")
#         return

#     tqdm.write(f"Dataset:{dataset_name} \t Method:{method}")
#     logging.info(f"Dataset:{dataset_name} \t Method:{method}")

#     if not os.path.exists(subspace_directory):
#         tqdm.write(f"The subspace directory does not exist.")
#         logging.info(f"The subspace directory does not exist.")
#         return

#     output_directory = f'{base_directory}/{method}/mds'
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)
#     results_df = pd.DataFrame(columns=['file_index', 'dimension', 'mds_error'])

#     # Iterate over all relevant files in the input directory

#     npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
#     if len(npy_files) == 0:
#         tqdm.write(f"No .npy files found in the directory.")
#         logging.info(f"No .npy files found in the directory.")
#         return

#     with tqdm(total=len(npy_files), unit=f"file", dynamic_ncols=True) as pbar:
#         for filename in npy_files:
#             file_path = os.path.join(input_directory, filename)
#             # Extract the file index from the filename
#             file_index = filename[2:-4]
#             X = np.load(file_path)
#             #print(f'MDS distortion: {method} \t file number:{file_index}' )
#             distance_matrix = compute_hyperbolic_distance_matrix(X)

#             if method in ['sfpca', 'pga']:
#                 # Load the subspace instance
#                 S = load_subspace_instance(subspace_directory, file_index)
#                 num_dimensions = np.shape(S.H)[1]
#             elif method in ['horopca', 'bsa']:
#                 Q, mu_ref = load_subspace_instance_qmu(subspace_directory, file_index)
#                 num_dimensions = np.shape(Q)[0] + 1

#             pbar.set_postfix({"file": filename})
#             pbar.update(1)
#             logging.info(f"Processed file: {filename}")

#             # Iterate through each dimension and compute the distance matrix and MDS error
#             for dimension in range(1, num_dimensions):
#                 inaccurate = False
#                 if method in ['sfpca','pga']:
#                     distance_matrix_d, inaccurate = compute_distance_matrix(X, S, dimension=dimension, method=method) 
#                 else:
#                     distance_matrix_d, inaccurate = compute_distance_matrix(X, Q, mu_ref, dimension, method)

#                 # Compute the MDS error
#                 mds_error = np.linalg.norm(distance_matrix - distance_matrix_d, 'fro') / np.linalg.norm(distance_matrix, 'fro')

#                 if inaccurate:
#                     mds_error = np.nan

#                 results_df = pd.concat([results_df, pd.DataFrame({
#                     'file_index': [file_index],
#                     'dimension': [dimension],
#                     'mds_error': [mds_error]
#                 })], ignore_index=True)
            
#     # Save the results to a CSV file
#     output_file_path = os.path.join(output_directory, 'distortions.csv')
#     results_df.to_csv(output_file_path, index=False)
# ###########################################################################
# ###########################################################################
# ###########################################################################
# def compute_quartet_results(dataset_name, method):
#     """
#     Compute Quartet Score results for a given dataset using the specified method.

#     Parameters:
#     dataset_name (str): Name of the dataset.
#     method (str): Method used ('sfpca', 'pga', 'horopca', or 'bsa').

#     Returns:
#     None
#     """
#     logging.info("Compute Quartet Scores.")
#     tqdm.write("Compute Quartet Scores.")

#     # Check if the provided method is valid
#     if method not in ['sfpca', 'pga', 'horopca', 'bsa']:
#         tqdm.write(f"Method {method} is not supported.")
#         logging.error(f"Method {method} is not supported.")
#         return

#     # Define directory paths
#     distance_file = 'hyperbolic_points'
#     base_directory = f'datasets/{dataset_name}'
#     subspace_directory = f'{base_directory}/{method}/subspaces'
#     input_directory = f'{base_directory}/{distance_file}'
    
#     if not os.path.exists(input_directory):
#         tqdm.write(f"The input directory does not exist.")
#         logging.info(f"The input directory does not exist.")
#         return

#     if not os.path.exists(subspace_directory):
#         tqdm.write(f"The subspace directory does not exist.")
#         logging.info(f"The subspace directory does not exist.")
#         return

#     output_directory = f'{base_directory}/{method}/quartet'
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)

#     logging.info(f"Dataset: {dataset_name} \t Method: {method}")
#     tqdm.write(f"Dataset: {dataset_name} \t Method: {method}")

#     # List all .npy files in the input directory that start with 'X'
#     npy_files = [f for f in sorted(os.listdir(input_directory)) if f.endswith(".npy") and f.startswith("X")]
#     if len(npy_files) == 0:
#         tqdm.write(f"No .npy files found in the directory.")
#         logging.info(f"No .npy files found in the directory.")
#         return

#     L = 10**5
#     results_df = pd.DataFrame(columns=['file_index', 'dimension', 'quartet_score'])

#     n = 0
#     # Iterate over all relevant files in the input directory
#     with tqdm(total=len(npy_files), unit=f"file", dynamic_ncols=True) as pbar:
#         for filename in npy_files:
#             n = n + 1
#             np.random.seed(n)

#             file_path = os.path.join(input_directory, filename)
#             # Extract the file index from the filename
#             file_index = filename[2:-4]
#             X = np.load(file_path)
#             N = np.shape(X)[1]

#             pbar.set_postfix({"file": filename})
#             pbar.update(1)
#             logging.info(f"Processed file: {filename}")

#             distance_matrix = compute_hyperbolic_distance_matrix(X)

#             random_matrix = np.random.uniform(0, 1, (2*L, 4))*N
#             integer_matrix = select_unique_combinations(random_matrix,L)
#             L_ = np.shape(integer_matrix)[0]

#             topology = []
#             for i in range(L_):
#                 index = integer_matrix[i,:]
#                 distance_matrix_i = distance_matrix[index,:].copy()
#                 distance_matrix_i = distance_matrix_i[:,index]
#                 topology.append( determine_best_topology(distance_matrix_i)) 
#             topology = np.array(topology)


#             if method in ['sfpca', 'pga']:
#                 # Load the subspace instance
#                 S = load_subspace_instance(subspace_directory, file_index)
#                 num_dimensions = np.shape(S.H)[1]
#             elif method in ['horopca', 'bsa']:
#                 Q, mu_ref = load_subspace_instance_qmu(subspace_directory, file_index)                
#                 num_dimensions = np.shape(Q)[0] + 1

#             # Iterate through each dimension and compute the distance matrix and MDS error
#             for dimension in range(1, num_dimensions):
#                 if method in ['sfpca','pga']:
#                     distance_matrix_d, inaccurate = compute_distance_matrix(X, S, dimension=dimension, method=method)  
#                 else:
#                     distance_matrix_d,inaccurate = compute_distance_matrix(X, Q, mu_ref, dimension, method)

#                 topology_d = []
#                 for i in range(L_):
#                     index = integer_matrix[i,:]
#                     distance_matrix_di = distance_matrix_d[index,:].copy()
#                     distance_matrix_di = distance_matrix_di[:,index]
#                     topology_d.append( determine_best_topology(distance_matrix_di)) 
#                 topology_d = np.array(topology_d)

#                 accuracy =  np.sum(np.abs(topology_d-topology) == 0)/L_

#                 if inaccurate:
#                     accuracy = np.nan
            
#                 results_df = pd.concat([results_df, pd.DataFrame({
#                     'file_index': [file_index],
#                     'dimension': [dimension],
#                     'quartet_score': [accuracy]
#                 })], ignore_index=True)
#             # # Save the results to a CSV file
#             output_file_path = os.path.join(output_directory, 'accuracies.csv')
#             results_df.to_csv(output_file_path, index=False)

#     # # Save the results to a CSV file
#     # output_file_path = os.path.join(output_directory, 'accuracies.csv')
#     # results_df.to_csv(output_file_path, index=False)






    # def translate(self, vector):
    #     # print(self.geometry,self.model == 'loid')
    #     if self.geometry == 'hyperbolic' and self.model == 'poincare':
    #         self.points = self.mobius_add(self.points, vector)
    #     elif self.geometry == 'hyperbolic' and self.model == 'loid':

    #         self.points = self.loid_add(self.points, vector)
    #         print('asd')
    #     else:
    #         if isinstance(self.points, np.ndarray):
    #             if len(vector) == self.dimension and np.linalg.norm(vector) < 1:
    #                 self.points += np.reshape(vector, (self.dimension, 1))
    #             else:
    #                 raise TypeError("Dimension or Norm of the translation vector is wrong.")
    #         elif isinstance(self.points, torch.Tensor):
    #             if len(vector) == self.dimension and torch.norm(vector) < 1:
    #                 self.points += vector.view(self.dimension, 1)
    #             else:
    #                 raise TypeError("Dimension or Norm of the translation vector is wrong.")
    #         else:
    #             raise TypeError("Points should be either a NumPy array or a PyTorch tensor.")








    # def get_lr_multiplier(loss_list, increase_factor=1.1, decrease_factor=0.5, window_size=50, max_increase_count=10):
#     if len(loss_list) < window_size:
#         return 1.0  # No change if there is not enough data

#     recent_losses = loss_list[-window_size:]
#     increase_count = sum(1 for x, y in zip(recent_losses, recent_losses[1:]) if y > x)

#     if increase_count > max_increase_count:
#         return decrease_factor
#     elif all(x > y for x, y in zip(recent_losses, recent_losses[1:])):
#         return increase_factor
#     else:
#         return 1.0  # No change

# class CustomLRScheduler:
#     def __init__(self, optimizer, increase_factor=1.1, decrease_factor=0.5,window_size=50, max_increase_count=10):
#         self.optimizer = optimizer
#         self.increase_factor = increase_factor
#         self.decrease_factor = decrease_factor
#         self.window_size = window_size
#         self.max_increase_count = max_increase_count  # Add this line
#         self.losses = []

#     def step(self, loss):
#         self.losses.append(loss.item())

#         # Keep only the latest `window_size` losses
#         if len(self.losses) > self.window_size:
#             self.losses.pop(0)

#         # If we don't have enough data yet, don't change the learning rate
#         if len(self.losses) < self.window_size:
#             return

#         increase_count = sum(1 for x, y in zip(self.losses, self.losses[1:]) if y > x)

#         if increase_count > self.max_increase_count:
#             self._adjust_learning_rate(self.decrease_factor)
#         else:
#             # If the loss is consistently decreasing
#             if all(x > y for x, y in zip(self.losses, self.losses[1:])):
#                 self._adjust_learning_rate(self.increase_factor)

#     def _adjust_learning_rate(self, factor):
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] *= factor
# ###########################################################################
# ###########################################################################
# ###########################################################################
# ###############################################
# def weight_exponent(epoch,distance_matrix):
#     epochs=2000
#     ratio_curv = 0.3

#     scale_range = distance_range(distance_matrix)
    
#     epochs_to_estimate_curvature = int(ratio_curv*epochs)
    
#     if epoch < epochs_to_estimate_curvature:
#         r = 0
#     else:
#         r =  -(epoch-epochs_to_estimate_curvature)/(epochs-1-epochs_to_estimate_curvature)

#     return r
# # ###############################################
# # ###############################################
# def give_me_params(epoch,learning_rate_init, distance_matrix):
#     epochs = 2000
    

#     scale_range = distance_range(distance_matrix)
#     scale = torch.tensor(1)

#     scale_learning = is_scale_learning(epoch)
#     r = weight_exponent(epoch,scale_range)

#     ratio_curv = 0.3
#     ratio_window = 0.025
#     num_bins = 200
    
#     window_size = int(ratio_window*epochs)
#     increase_factor = 1.001
#     decrease_factor = 0.98
#     max_increase_count = window_size//10
#     epochs_to_estimate_curvature = int(ratio_curv*epochs)


#     # custom_scheduler = CustomLRScheduler(optimizer_tangents, increase_factor, decrease_factor, window_size, max_increase_count)
#     return r, scale_learning


# def determine_best_topology(distance_matrix):
#     """
#     Determine the topology based on the four-point condition for a given 4x4 distance matrix.

#     Parameters:
#     distance_matrix (ndarray): A 4x4 distance matrix.

#     Returns:
#     int: The topology index (1, 2, or 3).
#     """

#     # Calculate the sums of the opposite pairs
#     sum_opposite_1 = distance_matrix[0, 1] + distance_matrix[2, 3]
#     sum_opposite_2 = distance_matrix[0, 2] + distance_matrix[1, 3]
#     sum_opposite_3 = distance_matrix[0, 3] + distance_matrix[1, 2]

#     # Find the minimum sum among the three calculated sums
#     minimum_sum = min(sum_opposite_1, sum_opposite_2, sum_opposite_3)

#     # Determine the topology index based on which sum is the minimum
#     if minimum_sum == sum_opposite_1:
#         topology_index = 1
#     elif minimum_sum == sum_opposite_2:
#         topology_index = 2
#     else:
#         topology_index = 3

#     return topology_index
# ###########################################################################
# ###########################################################################
# ###########################################################################
# def select_unique_combinations(random_matrix, L):
#     """
#     Select unique integer combinations from a random matrix, ensuring no duplicates in each row.

#     Parameters:
#     random_matrix (np.ndarray): Matrix of random values.
#     L (int): Number of unique rows to select.

#     Returns:
#     np.ndarray: Matrix with L unique rows.
#     """
#     # Convert the random matrix to integers
#     integer_matrix = random_matrix.astype(int)
    
#     # Sort each row to easily identify duplicates
#     sorted_matrix = np.sort(integer_matrix, axis=1)
    
#     # Identify rows with duplicate elements
#     has_duplicates = np.any(sorted_matrix[:, :-1] == sorted_matrix[:, 1:], axis=1)
    
#     # Filter out rows with duplicates
#     unique_matrix = integer_matrix[~has_duplicates]
    
#     # Select the first L unique rows
#     unique_matrix = unique_matrix[:L, :]
    
#     return unique_matrix
###########################################################################



# # Function to optimize the embedding
# def optimize_embedding(distance_matrix, 
#     epochs = 2000, 
#     dimension = 2,
#     learning_rate=None,
#     scale_learning=None,
#     weight_exponent=None,
#     initial_tangents=None, 
#     log_function=None
#     ):

#     n = distance_matrix.size(0)
#     if initial_tangents is None:
#         tangents = torch.rand(dimension,n, requires_grad=True)
#         tangents.data.mul_(0.01)
#     else:
#         tangents = initial_tangents.clone().detach().requires_grad_(True)

#     ##################################################################################
#     if learning_rate is None:
#         def learning_rate(epoch, total_epochs = 2000, loss_list = None):
#             learning_rate_init = 0.1
#             lr = learning_rate_init

#             ratio_window = 0.025
#             num_bins = 200
#             window_size = int(ratio_window*epochs)
#             increase_factor = 1.001
#             decrease_factor = 0.98
#             max_increase_count = window_size//10
#             ratio_curv = 0.3
#             epochs_to_estimate_curvature = int(ratio_curv*total_epochs)
            
#             if epoch < epochs_to_estimate_curvature:
#                 multipliers = []
#                 for i in range(1, len(loss_list) + 1):
#                     if i < window_size:
#                         multipliers.append(1.0)  # No change in learning rate if we don't have enough data for a point
#                         continue
#                     recent_losses = loss_list[i-window_size:i]
#                     increase_count = sum(1 for x, y in zip(recent_losses, recent_losses[1:]) if y > x)
#                     if increase_count > max_increase_count:
#                         multipliers.append(decrease_factor)  # Decrease learning rate for this point
#                     elif all(x > y for x, y in zip(recent_losses, recent_losses[1:])):
#                         multipliers.append(increase_factor)  # Increase learning rate for this point
#                     else:
#                         multipliers.append(1.0)  # No change in learning rate for this point
#                 lr = np.prod(multipliers)*learning_rate_init
#             else:
#                 multipliers = []
#                 loss_list = loss_list[0:epochs_to_estimate_curvature]
#                 for i in range(1, len(loss_list) + 1):
#                     if i < window_size:
#                         multipliers.append(1.0)  # No change in learning rate if we don't have enough data for a point
#                         continue
#                     recent_losses = loss_list[i-window_size:i]
#                     increase_count = sum(1 for x, y in zip(recent_losses, recent_losses[1:]) if y > x)
#                     if increase_count > max_increase_count:
#                         multipliers.append(decrease_factor)  # Decrease learning rate for this point
#                     elif all(x > y for x, y in zip(recent_losses, recent_losses[1:])):
#                         multipliers.append(increase_factor)  # Increase learning rate for this point
#                     else:
#                         multipliers.append(1.0)  # No change in learning rate for this point
#                 lr = np.prod(multipliers)*learning_rate_init


#                 r =  (epoch-epochs_to_estimate_curvature)/(epochs-1-epochs_to_estimate_curvature)
#                 p = 10**(-scale_range.item() / (epochs-epochs_to_estimate_curvature) )
#                 for _ in range(epochs_to_estimate_curvature, epoch):
#                     lr *= 10**(2*r*np.log10(p))
#             return lr
#     if scale_learning is None:
#         def scale_learning(epoch, total_epochs = 2000, loss_list = None):
#             ratio_curv = 0.3
#             epochs_to_estimate_curvature = int(ratio_curv*total_epochs)

#             if epoch < epochs_to_estimate_curvature:
#                 status = True
#             else:
#                 status = False
#             return status
#     if weight_exponent is None:
#         def weight_exponent(epoch, total_epochs = 2000, loss_list = None):
#             scale_range = distance_range(distance_matrix)
#             ratio_curv = 0.3
#             epochs_to_estimate_curvature = int(ratio_curv*epochs)
#             if epoch < epochs_to_estimate_curvature:
#                 r = 0
#             else:
#                 r =  -(epoch-epochs_to_estimate_curvature)/(epochs-1-epochs_to_estimate_curvature)
#             return r


#     # # Define optimizer
#     learning_rate_init = learning_rate(0, total_epochs = epochs, loss_list = [])
#     optimizer_tangents = optim.Adam([tangents], lr=learning_rate_init)
#     loss_list = []
#     scale = torch.tensor(1.0)
#     scale_grad_free = torch.tensor(scale.item())

#     for epoch in range(epochs):
#         optimizer_tangents.zero_grad()
#         status = scale_learning(epoch, total_epochs = 2000, loss_list = loss_list)
#         r = weight_exponent(epoch, total_epochs = 2000, loss_list = loss_list)
#         W = torch.pow(scale_grad_free*distance_matrix,r)
#         W.fill_diagonal_(1)
#         print(r, status,learning_rate(epoch, total_epochs = epochs, loss_list = loss_list))
#         if status:
#             loss, scale = cost_function(tangents, distance_matrix, scale_learning = True, W = W)
#             scale_grad_free = torch.tensor(scale.item())
#         else:
#             loss, scale = cost_function(tangents, distance_matrix, scale_learning = False, scale = scale_grad_free, W = W)

#         loss.backward()
#         loss_list.append(loss.item())
#         with torch.no_grad():
#             tangents.grad = torch.nan_to_num(tangents.grad, nan=0.0)
#         optimizer_tangents.step()

#         for param_group in optimizer_tangents.param_groups:
#                 param_group['lr'] = learning_rate(epoch, total_epochs = epochs, loss_list = loss_list)


#         current_lr = optimizer_tangents.param_groups[0]['lr']
#         message = f"Epoch {epoch}, Loss: {loss.item()}, Scale: {scale.item()}, Learning Rate: {current_lr}"
#         if log_function:
#             log_function(message)
#         else:
#             print(message)

#     return tangents,scale
# ###########################################################################



        
# class MultiTree():
#     def __init__(self, file_path: str):
#         """
#         Initialize a MultiTree object. Load trees from a file.
#         """
#         self.name = os.path.basename(file_path)
#         self.trees = self._load_trees(file_path)
#         self.metadata = {}

#     def copy(self) -> 'MultiTree':
#         """
#         Create a deep copy of the MultiTree object.
#         """
#         return copy.deepcopy(self)

#     def _load_trees(self, file_path: str) -> List[Tree]:
#         """
#         Load trees from a Newick file and return a list of Tree objects.
#         """
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {file_path} does not exist")
#         tree_list = []
#         try:
#             for idx, tree in enumerate(ts.read_tree_newick(file_path)):
#                 tree_list.append(Tree.from_contents(name=f'tree_{idx+1}', contents=tree))
#         except Exception as e:
#             raise ValueError(f"Failed to load trees from {file_path}: {e}")
#         return tree_list

#     def save_trees(self, file_path: str, format: str = 'newick') -> None:
#         """
#         Save all trees to a file in the specified format.
#         """
#         if format == 'newick':
#             with open(file_path, 'w') as f:
#                 for tree in self.trees:
#                     f.write(tree.contents.newick() + "\n")
#         else:
#             raise ValueError(f"Unsupported format: {format}")

#     def __iter__(self):
#         """
#         Return an iterator over the trees.
#         """
#         return iter(self.trees)

#     def __len__(self) -> int:
#         """
#         Return the number of trees.
#         """
#         return len(self.trees)

#     def __repr__(self) -> str:
#         """
#         Return a string representation of the MultiTree object.
#         """
#         return f"MultiTree({self.name}, {len(self.trees)} trees)"

#     def search_trees(self, query: str) -> List[Tree]:
#         """
#         Search for trees by name.
#         """
#         return [tree for tree in self.trees if query in tree.name]

#     def add_metadata(self, key: str, value: str) -> None:
#         """
#         Add metadata to the MultiTree object.
#         """
#         self.metadata[key] = value






# def hyperbolic_embedding(self, dimension: int, mode: str = 'naive', epochs: int = 2000, max_diameter: float = 10) -> 'HyperbolicSpace':
    #     """
    #     Naive embedding of the tree in a hyperbolic space.
    #     """
    #     scale_factor = max_diameter / self.tree.diameter()
    #     distance_matrix = self.tree.distance_matrix()
        
    #     distance_matrix *= scale_factor
    #     N = np.shape(distance_matrix)[0]

    #     self.log_info("Naive embedding of the tree in a hyperbolic space.")
    #     gramian = -np.cosh(distance_matrix)
    #     points = spaceform_pca_lib.lgram_to_points(dimension, gramian)
    #     for n in range(N):
    #         x = points[:, n]
    #         points[:, n] = spaceform_pca_lib.project_vector_to_hyperbolic_space(x)

    #     hyperbolic_space = embedding.HyperbolicSpace(model='loid', curvature=-(scale_factor**2))
    #     hyperbolic_space.points = points
    #     if mode != 'naive':    
    #         self.log_info("Precise Hyperbolic Embedding Step.")
    #         initial_tangents = spaceform_pca_lib.hyperbolic_log_torch( torch.tensor(points) )
    #         tangents, scale = spaceform_pca_lib.optimize_embedding(
    #             torch.tensor(distance_matrix),
    #             dimension = dimension,
    #             initial_tangents=initial_tangents, 
    #             epochs = epochs, 
    #             log_function=self.log_info
    #             )
    #         if scale.requires_grad:
    #             scale = scale.detach().numpy()
    #         else:
    #             scale = scale.numpy()
    #         hyperbolic_space.curvature *= np.abs(scale)**2
    #         hyperbolic_space.points = spaceform_pca_lib.hyperbolic_exponential_torch(tangents)
    #     return hyperbolic_space









    # class Tree:
#     def __init__(self, *source):
#         """
#         Initialize a Tree object. Load from a source, which can be:
#          -- a single string, giving the file path; name is set to base name by default
#          -- two inputs: a string (name) and a treeswift tree object (contents)
#         """
#         if len(source) == 1 and isinstance(source[0], str):
#             file_path = source[0]
#             self.name = os.path.basename(file_path)
#             self.contents = self._load_tree(file_path)
#         elif len(source) == 2 and isinstance(source[0], str) and isinstance(source[1], ts.Tree):
#             self.name = source[0]
#             self.contents = source[1]
#         else:
#             raise ValueError("Provide either a single file path as a string or two inputs: a string (name) and a treeswift tree object (contents).")
#         self.metadata = {}

#     @classmethod
#     def from_contents(cls, name: str, contents: ts.Tree) -> 'Tree':
#         """
#         Create a Tree object from given contents.
#         """
#         return cls(name=name, contents=contents)

#     def copy(self) -> 'Tree':
#         """
#         Create a deep copy of the Tree object.
#         """
#         return copy.deepcopy(self)

#     def _load_tree(self, file_path: str) -> ts.Tree:
#         """
#         Load a tree from a Newick file.
#         """
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {file_path} does not exist")
#         try:
#             return ts.read_tree_newick(file_path)
#         except Exception as e:
#             raise ValueError(f"Failed to load tree from {file_path}: {e}")

#     def save_tree(self, file_path: str, format: str = 'newick') -> None:
#         """
#         Save the tree to a file in the specified format.
#         """
#         if format == 'newick':
#             self.contents.write_tree_newick(file_path)
#         else:
#             raise ValueError(f"Unsupported format: {format}")

#     def __repr__(self) -> str:
#         """
#         Return a string representation of the Tree object.
#         """
#         return f"Tree({self.name})"

#     def terminal_names(self) -> List[str]:
#         """
#         Get the list of terminal (leaf) names in the tree.
#         """
#         return list(self.contents.labels(leaves=True, internal=False))

#     def distance_matrix(self) -> np.ndarray:
#         """
#         Compute and return the distance matrix for the tree.
#         """
#         labels = self.terminal_names()
#         distance_dict = self.contents.distance_matrix(leaf_labels=True)
#         label_to_index = {label: idx for idx, label in enumerate(labels)}
#         n = len(labels)
#         distance_matrix = np.zeros((n, n))

#         for label1, row in distance_dict.items():
#             i = label_to_index[label1]
#             for label2, distance in row.items():
#                 j = label_to_index[label2]
#                 distance_matrix[i, j] = distance
#         return distance_matrix

#     def diameter(self) -> float:
#         """
#         Compute and return the diameter of the tree.
#         """
#         return self.contents.diameter()

#     def normalize(self) -> None:
#         """
#         Normalize the branch lengths of the tree so that the diameter is 1.
#         """
#         diameter = self.diameter()
#         if not np.isclose(diameter, 0.0):
#             scale_factor = 1.0 / diameter
#             for node in self.contents.traverse_postorder():
#                 if node.get_edge_length() is not None:
#                     node.set_edge_length(node.get_edge_length() * scale_factor)









# class MultiTree(Collection):
#     def __init__(self, *source: Union[str, Set[ts.Tree], str]):
#         """
#         Initialize a MultiTree object. Load trees from a source, which can be:
#          -- a single string, giving the file path; name is set to base name by default
#          -- a set of ts.Tree objects and a string (name)
#         """
#         if len(source) == 1 and isinstance(source[0], str):
#             file_path = source[0]
#             self.name = os.path.basename(file_path)
#             self.trees = self._load_trees(file_path)
#         elif len(source) == 2 and isinstance(source[0], str) and isinstance(source[1], Set):
#             self.name = source[0]
#             self.trees = [Tree(tree.name, tree.contents) for tree in source[1]]
#         else:
#             raise ValueError("Provide either a single file path as a string or two inputs: a string (name) and a set of ts.Tree objects (contents).")
#         self.metadata = {}

#     def copy(self) -> 'MultiTree':
#         """
#         Create a deep copy of the MultiTree object.
#         """
#         return copy.deepcopy(self)

#     def _load_trees(self, file_path: str) -> Collection[Tree]:
#         """
#         Load trees from a Newick file and return a list of Tree objects.
#         """
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file {file_path} does not exist")
#         tree_list = []
#         try:
#             for idx, tree in enumerate(ts.read_tree_newick(file_path)):
#                 tree_list.append(Tree(f'tree_{idx+1}', tree))
#         except Exception as e:
#             raise ValueError(f"Failed to load trees from {file_path}: {e}")
#         return tree_list

#     def save_trees(self, file_path: str, format: str = 'newick') -> None:
#         """
#         Save all trees to a file in the specified format.
#         """
#         if format == 'newick':
#             with open(file_path, 'w') as f:
#                 for tree in self.trees:
#                     f.write(tree.contents.newick() + "\n")
#         else:
#             raise ValueError(f"Unsupported format: {format}")

#     def distance_matrix(self, 
#                            func: Callable[[np.ndarray], float] = np.nanmean, 
#                            confidence: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
#         """
#         Compute the distance matrix of each individual tree and compute the element-wise aggregate
#         (mean, median, max, min) of them. The aggregate function can be specified as an input.
        
#         Additionally, return a confidence matrix indicating the ratio of non-NaN values at each element.

#         Parameters:
#         func (Callable[[np.ndarray], float]): Function to compute the aggregate. Default is np.nanmean.
#         confidence (bool): If True, also return the confidence matrix. Default is False.

#         Returns:
#         Tuple[np.ndarray, Optional[np.ndarray]]:
#             - The aggregated distance matrix.
#             - (Optional) Confidence matrix indicating the ratio of non-NaN values.
#         """
#         if not self.trees:
#             raise ValueError("No trees available to compute distance matrices")

#         distance_dicts = []
#         all_labels = set()
#         for tree in self.trees:
#             labels = tree.contents.labels(leaves=True, internal=False)
#             distance_dict = tree.contents.distance_matrix(leaf_labels=True)
#             distance_dicts.append(distance_dict)
#             all_labels.update(labels)
        
#         all_labels = sorted(all_labels)
#         label_to_index = {label: idx for idx, label in enumerate(all_labels)}
#         n = len(all_labels)
        
#         distance_matrices = []
#         count_matrices = []
#         for distance_dict in distance_dicts:
#             distance_matrix = np.full((n, n), np.nan)
#             count_matrix = np.zeros((n, n))  # Matrix to count non-NaN entries
#             for label1, row in distance_dict.items():
#                 i = label_to_index[label1]
#                 for label2, distance in row.items():
#                     j = label_to_index[label2]
#                     distance_matrix[i, j] = distance
#                     count_matrix[i, j] += 1
#             np.fill_diagonal(distance_matrix, 0)  # Fill diagonal with 0
#             distance_matrices.append(distance_matrix)
#             count_matrices.append(count_matrix)
        
#         stacked_matrices = np.stack(distance_matrices)
#         stacked_counts = np.stack(count_matrices)

#         # Apply the aggregation function element-wise, excluding NaNs
#         aggregated_matrix = func(stacked_matrices, axis=0)

#         # Compute confidence matrix
#         # Confidence is the ratio of non-NaN counts to the number of trees
#         confidence_matrix = np.sum(stacked_counts > 0, axis=0) / len(self.trees)

#         if confidence:
#             return aggregated_matrix, confidence_matrix
#         else:
#             return aggregated_matrix


#     def search_trees(self, query: str) -> List[Tree]:
#         """
#         Search for trees by name.
#         """
#         return [tree for tree in self.trees if query in tree.name]

#     def add_metadata(self, key: str, value: str) -> None:
#         """
#         Add metadata to the MultiTree object.
#         """
#         self.metadata[key] = value

#     def __iter__(self) -> Collection[Tree]:
#         """
#         Return an iterator over the trees.
#         """
#         return iter(self.trees)

#     def __len__(self) -> int:
#         """
#         Return the number of trees.
#         """
#         return len(self.trees)

#     def __contains__(self, item) -> bool:
#         """
#         Check if an item is in the collection.
#         """
#         return item in self.trees

#     def __repr__(self) -> str:
#         """
#         Return a string representation of the MultiTree object.
#         """
#         return f"MultiTree({self.name}, {len(self.trees)} trees)"




# def load_npy_files(directory):
                #     npy_files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
                #     matrices = [np.load(os.path.join(directory, f)) for f in npy_files]
                #     return npy_files, matrices

                # def plot_heatmaps(re_matrix, distance_matrix, epoch, save_path):
                #     # Create a mask for the diagonal
                #     mask = np.eye(re_matrix.shape[0], dtype=bool)
                    
                #     # Set non-positive values to a small positive value to avoid invalid log10 results
                #     log_eps = np.log10(np.finfo(float).eps)
                    
                #     with np.errstate(divide='ignore', invalid='ignore'):
                #         log10_re_matrix = np.log10(re_matrix)
                #     with np.errstate(divide='ignore', invalid='ignore'):
                #         log10_distance_matrix = np.log10(distance_matrix)
                    
                #     log10_re_matrix[np.isinf(log10_re_matrix) | np.isnan(log10_re_matrix)] = log_eps    
                #     log10_distance_matrix[np.isinf(log10_distance_matrix) | np.isnan(log10_distance_matrix)] = log_eps
                    
                #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                #     max_log_re = 4
                #     sns.heatmap(log10_re_matrix, mask=mask, ax=axes[0], cmap='viridis', cbar_kws={'label': 'log10(RE)'}, vmin=log_eps, vmax=max_log_re)
                #     # sns.heatmap(log10_re_matrix, mask=mask, ax=axes[0], cmap=cmap, cbar_kws={'label': 'log10(RE)'}, vmin=log_eps, vmax=max_log_re)
                #     axes[0].set_title(f'RE Matrix (Epoch {epoch})')
                    
                #     sns.heatmap(log10_distance_matrix, mask=mask, ax=axes[1], cmap='viridis', cbar_kws={'label': 'log10(Distance)'})
                #     axes[1].set_title('Distance Matrix')
                    
                #     plt.tight_layout()
                #     plt.savefig(save_path, dpi=300)
                #     plt.close()



import numpy as np
from scipy.linalg import sqrtm

class EmbeddingSpace:
    def __init__(self, model, name, points=None):
        self.name = name
        self.model = model
        self._points = points if points is not None else np.empty((0, 0))
        self.update_dimensions()
    ##########################################
    @property
    def points(self):
        return self._points
    ##########################################
    @points.setter
    def points(self, value):
        self._points = value
        self.update_dimensions()
    ##########################################
    def update_dimensions(self):
        self.n_points = self._points.shape[1] if self._points.size else 0
        self.dimension = self._points.shape[0] if self._points.size else 0
    ##########################################
    def translate(self, vector):
        raise NotImplementedError("This method should be implemented by subclasses.")
    ##########################################
    def rotate(self, R):
        raise NotImplementedError("This method should be implemented by subclasses.")
    ##########################################
    def center(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    ##########################################
    def centroid(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    ##########################################
    def distance_matrix(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
class HyperbolicSpace(EmbeddingSpace):
    def __init__(self, model='loid', curvature=-1, points=None):
        super().__init__('hyperbolic', points)
        self.curvature = curvature
        self.model = model
        self.update_dimensions()
    ##########################################
    def __repr__(self):
        return f"PointSet({self.points}, geometry=hyperbolic('{self.model}'), curvature={self.curvature})"
        ##########################################
    def update_dimensions(self):
        super().update_dimensions()
        if self.model == 'loid':
            self.dimension = self.points.shape[0] - 1 if self.points.size else 0
    ##########################################
    def mobius_add(self, x, y):
        if isinstance(x, np.ndarray):
            norm_x_sq = np.sum(x ** 2, axis=0, keepdims=True)
            norm_y_sq = np.sum(y ** 2)
            dot_product = 2 * np.dot(x.T, y)
            denominator = 1 + dot_product + norm_x_sq * norm_y_sq
            numerator = x * (1 + dot_product + norm_y_sq) + y.reshape(-1, 1) * (1 - norm_x_sq)
            return numerator / denominator
        elif isinstance(x, torch.Tensor):
            norm_x_sq = torch.sum(x ** 2, dim=0, keepdim=True)
            norm_y_sq = torch.sum(y ** 2)
            dot_product = 2 * torch.matmul(x.T, y)
            denominator = 1 + dot_product + norm_x_sq * norm_y_sq
            numerator = x * (1 + dot_product + norm_y_sq) + y.view(-1, 1) * (1 - norm_x_sq)
            return numerator / denominator
        else:
            raise TypeError("Points should be either a NumPy array or a PyTorch tensor.")
    ##########################################
    def loid_add(self, x, y):
        if isinstance(x, np.ndarray):
            y = np.asarray(y).reshape(-1, 1)
            hyperbolic_norm = -y[0]**2 + np.sum(y[1:]**2)
            if not np.isclose(hyperbolic_norm, -1):
                raise ValueError("The hyperbolic norm of the vector must be -1")
            
            D = len(y)-1
            b = y[1:]
            norm_b = np.linalg.norm(b)
            I = np.eye(D)
            
            Ry = np.zeros((D+1,D+1))
            Ry[0,0] = np.sqrt(1+norm_b**2)
            Ry[0,1:] = b.reshape(1, -1).flatten()
            Ry[1:,0] = b.reshape(-1, 1).flatten()
            Ry[1:,1:] = sqrtm(I+np.outer(b, b.T) )
            return Ry @ x
        elif isinstance(x, torch.Tensor):
            y = y.view(-1, 1)
            hyperbolic_norm = -y[0]**2 + torch.sum(y[1:]**2)
            if not torch.isclose(hyperbolic_norm, torch.tensor(-1.0)):
                raise ValueError("The hyperbolic norm of the vector must be -1")
            D = len(y) - 1
            b = y[1:]
            norm_b = torch.norm(b)
            I = torch.eye(D)
            
            Ry = torch.zeros((D+1, D+1))
            Ry[0, 0] = torch.sqrt(1 + norm_b**2)
            Ry[0, 1:] = b.view(1, -1).flatten()
            Ry[1:, 0] = b.view(-1, 1).flatten()
            Ry[1:, 1:] = torch.linalg.sqrtm(I + torch.outer(b, b.T))
            
            return Ry @ x
        else:
            raise TypeError("Input x must be either a NumPy array or a PyTorch tensor.")
    ##########################################
    def translate(self, vector):
        if self.model == 'poincare':
            self.points = self.mobius_add(self.points, vector)
        elif self.model == 'loid':
            self.points = self.loid_add(self.points, vector)
        else:
            raise ValueError("Invalid model for hyperbolic space.")
    ##########################################
    def rotate(self, R):
        if self.model == 'loid':
            if isinstance(R, np.ndarray):
                D = R.shape[0]
                new_matrix = np.zeros((D + 1, D + 1))
                new_matrix[0, 0] = 1
                new_matrix[0, 1:] = np.zeros(D)
                new_matrix[1:, 0] = np.zeros(D)
                new_matrix[1:, 1:] = R
                self.points = new_matrix @ self.points
            elif isinstance(R, torch.Tensor):
                D = R.size(0)
                new_matrix = torch.zeros((D + 1, D + 1))
                new_matrix[0, 0] = 1
                new_matrix[0, 1:] = torch.zeros(D)
                new_matrix[1:, 0] = torch.zeros(D)
                new_matrix[1:, 1:] = R
                self.points = new_matrix @ self.points
            else:
                raise TypeError("Input R must be a NumPy array or a PyTorch tensor.")
        else:
            self.points = R @ self.points
    ##########################################
    def center(self):
        centroid = self.centroid()
        if self.model == 'loid':
            self.translate(-centroid)
        else:
            self.translate(-centroid)
    ##########################################
    def centroid(self):
        if self.model == 'loid':
            avg_point = np.mean(self.points[1:], axis=1)
            norm_avg_point = np.linalg.norm(avg_point)
            centroid = np.zeros(self.dimension + 1)
            centroid[0] = np.sqrt(1 + norm_avg_point**2)
            centroid[1:] = avg_point
            return centroid
        elif self.model == 'poincare':
            # Temporarily switch to 'loid' to compute the centroid
            self.switch_model('loid')
            avg_point_loid = np.mean(self.points[1:], axis=1)
            norm_avg_point_loid = np.linalg.norm(avg_point_loid)
            centroid_loid = np.zeros(self.dimension + 1)
            centroid_loid[0] = np.sqrt(1 + norm_avg_point_loid**2)
            centroid_loid[1:] = avg_point_loid
            
            # Transform centroid from 'loid' to 'poincare'
            x1 = centroid_loid[0]
            bar_x = centroid_loid[1:]
            centroid_poincare = bar_x / (x1 + 1)
            
            # Switch back to 'poincare'
            self.switch_model('poincare')
            return centroid_poincare
        else:
            raise ValueError("Unknown model.")
    ##########################################
    def switch_model(self, new_model):
        if new_model == self.model:
            return

        if new_model == 'loid':
            if self.points.size == 0:
                raise ValueError("No points to switch model.")
            
            norm_points = np.linalg.norm(self.points, axis=0)
            if norm_points.size == 0:
                raise ValueError("Points array is empty.")
            new_points = np.zeros((self.points.shape[0] + 1, self.points.shape[1]))
            new_points[0] = (1 + norm_points**2) / (1 - norm_points**2)
            new_points[1:] = (2 * self.points) / (1 - norm_points**2)
            self.points = new_points
            self.model = new_model
            self.dimension = self.points.shape[0] - 1
        elif new_model == 'poincare':
            # Switch from Loid to Poincare
            if self.points.size == 0:
                raise ValueError("No points to switch model.")
            x1 = self.points[0]
            bar_x = self.points[1:]
            self.points = bar_x / (x1 + 1)
            self.model = new_model
            self.dimension = self.points.shape[0]
        else:
            raise ValueError("Invalid model for hyperbolic space.")
        self.update_dimensions()
    ##########################################
    def distance_matrix(self): 
        X = self.points
        D = self.dimension
        scale = 1/ np.sqrt(-self.curvature)
        J = np.eye(D+1)
        J[0,0] = -1

        G = np.matmul(np.matmul(X.T,J),X)
        G[G>= -1] = -1
        distance_matrix = scale * np.arccosh(-G)
        return distance_matrix
    ##########################################


