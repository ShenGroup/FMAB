# Federated Multi-armed Bandits

This is the package of codes and datasets used in paper ''Federated Multi-armed Bandits'', which is accepted to AAAI 2021. 

The files ''Fed1_UCB_CR.py'', ''Fed2_UCB_CR.py'' and  ''Fed2_UCB_CR_short.py'' are for the simulations of cognitive radio systems with the synthetic datasets. ''Fed1_UCB_RS.py'' and ''Fed2_UCB_RS.py'' are for the simulations of recommender systems with the MovieLens datasets. The synthetic datasets are generated in the corresponding codes and the preprocessed MovieLens datasets are in the file ''movielens_norm_100.npy''. The original MovieLens datasets can be downloaded [here](https://grouplens.org/datasets/hetrec-2011/) and the preprocessing steps are specified in the paper.

## Dependencies

The original codes are written with Python 3.7, and the needed packages are ''numpy 1.18.1'' and ''matplotlib 3.1.3''.

## Results

The performance of Fed1-UCB algorithm with the synthetic datasets as shown in Fig. 3 can be get by directly running the file ''Fed1_UCB_CR.py''. The default setting is for $f(p)=\lceil10\log(T)\rceil$ and $M=5$ with communication loss. To ignore the communication loss, comment out the line of computing it, which is labelled in the code. Results with other choices of $M$ and $f(p)$ under different bandit environments can also be get by changing the corresponding parameters in the code.

The performance of Fed2-UCB algorithm with the synthetic datasets as shown in Fig. 4 can be get by directly running the file ''Fed2_UCB_CR.py''. The default setting is for $f(p)=100$ and $g(p)=2^p$ with communication loss. Similar changes can be made as above to get the other results.

The performance of Fed2-UCB algorithm with the synthetic datasets and a reduced horizon as shown in Fig. 5 can be get by directly running the file ''Fed2_UCB_CR_short.py''. The default setting is for $f(p)=50$ and $g(p)=2^p$ with communication loss. Similar changes can be made as above to get the other results.

The performance of Fed1-UCB algorithm with the real-world datasets as shown in Fig. 6 can be get by directly running the file ''Fed1_UCB_RS.py''. The default setting for Fed1-UCB is $f(p)=\lceil10\log(T)\rceil$ with all available clients, which is for the curve labelled as ''Fed1-UCB, full'' in Fig. 5. The curve labelled as ''Fed1-UCB, $M=200$'' can be get by changing $M=200$ in the code. The results for Fed2-UCB can be get by directly running the file ''Fed2_UCB_RS.py'', which is set for $f(p)=200$ and $g(p)=2^p$ by default.
