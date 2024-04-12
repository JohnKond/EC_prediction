<center>
  
<h1> <b>Correlation Analysis of Salinity Indices & Soil Electrical Conductivity at a 1-meter Depth</b> </h1>


  
<b>Author</b>      : Ioannis Kontogiorgakis <br>
<b>Associates</b> : Nikolaos Gerarchakis <br>
<b>Date</b>             : February 9, 2023 <br>

<br>

![](https://drive.google.com/uc?export=view&id=13qxDn2dk5xE6U4tZW-zmKBjo95RnTg44)

<h3> <b>Abstract</b> </h3>

Despite the importance of soil electrical conductivity (EC) in precision agriculture, traditional measurement methods are labor-intensive
and time-consuming, requiring farmers to manually survey vast fields. Moreover, the reliance on ground-based measurements limits the scalability and efficiency of soil monitoring efforts. In this project, we aim
to propose a new method of measuring Electrical Conductivity by utilizing soil salinity indices.

<h3> <b>Dataset</b> </h3>

The salinity measurements were acquired from <b>Sentinel-2</b> satellite images via <b>Google Earth</b>. It consists of soil, atmospheric measurements of a specific field in Heraklion, Greece conducted in 2 different dates:

<ul>
<li>salinity indices of 01-09-2023</li>
<li>salinity indices of 02-09-2023</li>
</ul>

In order to create some ground truth data about our target variable "EC", we measured the soil electrical conductivity manually using with a soil EC-meter :
<ul>
<li> EC measurements of 01-09-2023</li>
</ul>


<h2><b> Outcome </b></h2>

<ul>
<li> We introduced a novel approach for measuring Soil Electrical Conductivity that avoids the need for manual tools.
<li>Developed a robust XGB model with an $r^2$ score of <b>0.9788</b>, showcasing its ability to predict Soil Electrical Conductivity accurately, as shown from the final model visualization.</li>
<li>Achieved a low RMSE of <b>4.6980</b> and MSE of <b>22.0712</b>, indicating the model's precision in estimating EC values.</li>
<li>The model was trained efficiently in just <b>6.06</b> seconds, demonstrating its scalability for large datasets.</li>
</ul>
