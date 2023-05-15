#!/bin/bash

NO_CHANGES_MD5=$(perl script/edit_config_ini.pl --config_path=test/projects/allparamtypes/config.ini | md5sum | sed -e 's/ .*//')

if [[ "$NO_CHANGES_MD5" == "d3de3e9d04fecaef7e0611cfc2c7ace9" ]]; then
        echo "No changes ok";
else
        echo "No changes failed"
        exit 1
fi

DELETED_DIMENSION_MD5=$(perl script/edit_config_ini.pl --config_path=test/projects/allparamtypes/config.ini --delete_dimension=lognormal | md5sum | sed -e 's/ .*//')
if [[ "$DELETED_DIMENSION_MD5" == "e1ccb0d5e5cbc5cdc1f0d5f0a4247bb1" ]]; then
        echo "Deleted dimension ok";
else
        echo "Deleted dimension failed"
        exit 2
fi

exit 0
