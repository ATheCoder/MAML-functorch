import torch

def generate_random_task(device):
    example_support = torch.rand((25, 3, 84, 84)).to(device)
    supprot_labels = torch.randint(0, 2, (25, 5)).float().to(device)

    example_query = torch.rand((50, 3, 84, 84)).to(device)
    query_labels = torch.randint(0, 2, (50, 5)).float().to(device)
    
    return example_support, supprot_labels, example_query, query_labels
    

def generate_random_batch(batch_size, device):
    support_samples = []
    support_labels = []
    
    query_samples = []
    query_labels = []
    
    
    for i in range(batch_size):
        example_support, s_l, example_query, q_l = generate_random_task(device)
        
        support_samples.append(example_support)
        support_labels.append(s_l)
        
        query_samples.append(example_query)
        query_labels.append(q_l)
    
    support_samples = torch.stack(support_samples)
    support_labels = torch.stack(support_labels)
    
    query_samples = torch.stack(query_samples)
    query_labels = torch.stack(query_labels)
    
    
    return support_samples, support_labels, query_samples, query_labels