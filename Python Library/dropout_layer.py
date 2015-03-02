

    if random_seed == None:                                                        
              rand = T.shared_randomstreams.RandomStreams(np.random.randint(999999))       
                  else:                                                                          
                            rand = T.shared_randomstreams.RandomStreams(random_seed)                     
                                if rms_injection_rate == None:                                                 
                                          rms_injection_rate = 1 - rms_decay_rate                                      
                                              masked_train_set_X = train_set_X * rand.binomial(n=1, p=1-dropout, size=(batch_size, self.num_features)).eval()
                                                                                                                                 
                                                                                                                                     dropout_layer = dropout                                                        
                                                                                                                                     
