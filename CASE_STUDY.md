# Case Study: Air Quality Forecasting from Weather Data

## The problem

Fine particulate pollution is one of the largest environmental health risks in the world, and the cities that suffer most from it are frequently the least equipped to monitor it. A reference grade PM2.5 monitoring station costs tens of thousands of dollars to install and requires ongoing calibration and maintenance, so coverage tends to be thin outside major metropolitan centres. Residents of smaller cities and rural districts often have no local reading at all, and no way to know whether tomorrow warrants keeping children indoors.

Weather forecasting infrastructure, by contrast, is close to universal. Temperature, dew point, pressure, and wind are predicted several days ahead for essentially every populated area on earth, at no cost to the end user. This project asked whether that existing, free infrastructure could be repurposed to produce air quality warnings where pollution sensors do not exist.

## What was built and what it showed

A regression model trained on five years of hourly Beijing data, pairing embassy PM2.5 readings with airport meteorological records. Three algorithms were compared, and gradient boosting performed best, explaining roughly half the variation in pollution levels on a full year of held out data and reducing prediction error by about 30 percent against a naive baseline.

The most useful finding was not the headline accuracy but what drove it. The single most important predictor was not a weather reading at all: it was an engineered flag marking Beijing's coal fired district heating season, which runs on a fixed schedule from mid November to mid March. A feature built from knowledge about how the city operates outperformed every raw meteorological measurement available. That is a reminder with fairly general application, that domain understanding often contributes more than additional data or a more sophisticated algorithm.

## Real world value, and the limits of it

The practical case is straightforward. A public health authority in a city without monitoring infrastructure could pair this approach with a standard weather forecast to issue advisory warnings a day or two ahead, at effectively zero marginal cost. Schools could make informed decisions about outdoor activities. People with asthma or cardiovascular conditions could plan around bad days rather than discover them. None of this replaces a monitoring network, but it is considerably better than the nothing that many communities currently have.

The honest limitation is equally important, and it is the point I would lead with in front of a stakeholder rather than bury. Weather does not create pollution; it disperses or traps pollution that emissions have already produced. Wind clears particulates out of a city, stagnant air lets them accumulate, and a temperature inversion holds them near ground level. But the model has no visibility of traffic volume, industrial output, construction dust, or crop burning, which is where the pollution actually comes from. This is why roughly half the variation remains unexplained, and why no additional weather data would recover it.

That limitation shapes where the model can responsibly be used. It is suitable for advisory guidance, for planning, and for filling a gap where the alternative is no information. It is not suitable as a regulatory instrument or as a substitute for measurement, and it consistently under predicts the severe episodes that matter most, because multi day smog accumulation cannot be inferred from a single hour of weather. Deploying it without stating that clearly would risk giving people false reassurance on exactly the days they most need a warning, which would be worse than providing nothing at all.
