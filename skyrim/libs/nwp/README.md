# ECMWF Forecasts S3 Structure

This README provides information about the folder structure of ECMWF forecasts stored in the AWS S3 bucket `ecmwf-forecasts`.

## Folder Structure Evolution

The folder structure has evolved over time. Here's a breakdown of the changes:

### Current Structure (2024-02-29 onwards)

```
s3://ecmwf-forecasts/<date>/<time>z/
├── aifs/
│   └── 0p25/
│       └── oper/
└── ifs/
    ├── 0p25/
    │   ├── enfo/
    │   ├── oper/
    │   ├── waef/
    │   └── wave/
    └── 0p4-beta/
```

### Intermediate Structure (2024-02-01 to 2024-02-28)

```
s3://ecmwf-forecasts/<date>/<time>z/
├── 0p25/
│   ├── enfo/
│   ├── oper/
│   ├── waef/
│   └── wave/
└── 0p4-beta/
    ├── enfo/
    ├── oper/
    ├── waef/
    └── wave/
```

### Previous Structure (2023-01-18 to 2024-01-31)

```
s3://ecmwf-forecasts/<date>/<time>z/
└── 0p4-beta/
    ├── enfo/
    ├── oper/
    ├── waef/
    └── wave/
```

## Notes

- `<date>` format: YYYYMMDD
- `<time>` format: 00, 06, 12, 18
- The bucket can be accessed without authentication using the `--no-sign` option with the AWS CLI.
- The `aifs` folder contains the Artificial Intelligence/Integrated Forecasting System (AIFS), while the `ifs` folder contains the Integrated Forecasting System data.
- Both `0p25` and `0p4-beta` resolutions contain the same set of subfolders: enfo, oper, waef, and wave.

## Additional Information
* [AIFS product specs](https://www.ecmwf.int/en/forecasts/dataset/aifs-machine-learning-data)
    
