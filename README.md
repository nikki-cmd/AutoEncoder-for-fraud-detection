You can find the dataset on Google Drive by following this link:
https://drive.google.com/drive/folders/1JB0mRbIONNkB6F4T6G9-GWzzf037Er-B?usp=drive_link

In this work I've created an Autoencoder model for anomaly detection in transactions. 
Anomaly detection using autoencoders typically relies on using the reconstruction loss, often the mean squared error (MSE), as a proxy for “anomalousness”. AEs are often employed to model the data distribution of normal inputs and subsequently identify anomalous, out-of-distribution inputs by high reconstruction error or low likelihood, respectively. However, AEs may generalize and achieve small reconstruction errors on abnormal inputs.
The underlying assumption is that anomalies are harder to reconstruct, and will therefore have a higher reconstruction loss. 
