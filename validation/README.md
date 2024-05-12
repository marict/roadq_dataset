# Preparing the validation data

The validation data was already prepared and saved in [validation_samples.csv](https://drive.google.com/drive/folders/1MgrCHc3jqs1LenSslTPpn6vd-lByKsB0)

**Eastimated date of PCI measurements: 2024-04-23**

## PCI Measurements

Download the measurements from [seattle_streets_min.csv](https://docs.google.com/spreadsheets/d/1X_VBSyIlwAPGk2n_75YhInCifkS7YVr5vAFerM8fBMQ/edit#gid=1487968714) to the same directory as `validation_data.py`.

The PCI measurements were collected from [seattlecitygis.maps.arcgis.com](https://seattlecitygis.maps.arcgis.com/home/item.html?id=d716876edede4fbd9c614978683b1c91&view=list&sortOrder=desc&sortField=defaultFSOrder#overview) and processed with the command below to remove unnecessary columns.

```bash
prepare-seattle-street.py prepare-seattle-street
```

## Map information

Download the osm data.

```bash
prepare-seattle-street.py \
    --east=-122.351294 \
    --west=-122.341117 \
    --north=47.632294 \
    --south=47.627125
```

## Cross generated validation samples

```bash
prepare-seattle-street.py generate-validation-data
```