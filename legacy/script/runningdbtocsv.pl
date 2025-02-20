#!/usr/bin/perl

$|++;

use strict;
use warnings;
use Data::Dumper;

use IO::Handle;
STDERR->autoflush(1);

use lib './perllib';
use Env::Modify;

modify_system(q"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib64/");

our %options = (
        debug => 0,
        project => '',
        int => 0,
        seperator => ';',
        projectdir => './projects',
        filename => get_random_nonexistant_file(),
        stdout => 1
);

use OmniOptFunctions;

analyze_args(@ARGV);

main();

sub get_random_nonexistant_file {
        my $rand_filename = ".".rand().".txt";
        while (-e $rand_filename) {
            $rand_filename = ".".rand().".txt";
        }
        return $rand_filename;
}

sub main {
    p(1, "Started main script");
    my $projectfolder = get_project_folder($options{project});

    p(2, "Loading modules");
    modules_load(
            "release/23.04", 
            "MongoDB/4.0.3", 
	    "GCC/11.3.0",
	    "OpenMPI/4.1.4",
            "Hyperopt/0.2.7"
    );
    p(10, "Loading modules");

    modify_system('PYTHONPATH=$HOME/.local/lib/python3.7/site-packages:/software/haswell/Python/3.7.4-GCCcore-8.3.0/lib/python3.7/site-packages:$PYTHONPATH');
    warn "If the module past is missing, install the package future via pip3 install --user future\n";
    warn "If the module pymongo is missing, install the package future via pip3 install --user pymongo\n";

    my $command = qq#python3 script/dbtocsv.py #.($options{int} ? '--int=1' : '').qq# --project=$options{project} --seperator="$options{seperator}" #;

    if($options{filename}) {
        $command .= qq# --filename="$options{filename}"#;
    }

    if(-d $options{projectdir}) {
        $command .= " --projectdir=".$options{projectdir}." ";
    }
    
    my $command_end_mongodb = undef;

    my $slurm_id_project = get_id_of_project();
    if(defined $slurm_id_project) {
            p(20, "Found Slurm-ID $slurm_id_project");
            my $ipfilefolder = "$projectfolder/ipfiles";
            if(-d $ipfilefolder) {
                    my ($dbip, $dbport) = map { read_file("$ipfilefolder/$_") } ("mongodbserverip-$slurm_id_project", "mongodbportfile-$slurm_id_project");
                    chomp $dbip;
                    chomp $dbport;

                    warn "Found the following ip:port: $dbip:$dbport\n";
                    $command .= qq# --mongodbmachine=$dbip --mongodbport=$dbport #;
            } else {
                    die "$ipfilefolder not found";
            }
    } else {
            p(20, "Found no Slurm-ID, starting database");
            warn "The running slurm job for the project `$options{project}` could not be found. Is it running and has the same slurm name? If not, a new database instance will be started and this warning can be ignored.";

            my $lockfile = "$projectfolder/mongodb/mongod.lock";

            if(!-e $lockfile) {
                    $command_end_mongodb = qq#python3 script/endmongodb.py --project=$options{project} #;
                    if($options{projectdir}) {
                            $command_end_mongodb .= qq#--projectdir=$options{projectdir}#;
                    }
            } else {
                    die "There is a lockfile `$lockfile`, that means the server was either not shut down correctly or it is running in another job. Better not doing anything automatically.\n";
            }
    }

    if($command) {
            my $stdout = modify_system($command);
            if($options{stdout}) {
                    if(-e $options{filename}) {
                            warn read_file($options{filename});
                            unlink $options{filename};
                    } else {
                            warn "$options{filename} could not be found";
                    }
            }
    } else {
            warn "ERROR: \$command is empty";
    }

    if($command_end_mongodb) {
            modify_system($command_end_mongodb);
    }
}

sub analyze_args {
        my @args = @_;

        foreach (@args) {
                if(m#^--debug$#) {
                        $options{debug} = 1;
                } elsif (m#^--project=(.*)$#) {
                        $options{project} = $1;
                } elsif (m#^--int$#) {
                        $options{int} = 1;
                } elsif (m#^--seperator=(.+)$#) {
                        $options{seperator} = $1;
                } elsif (m#^--nostdout=(.+)$#) {
                        $options{stdout} = 0;
                } elsif (m#^--filename=(.+)$#) {
                        $options{filename} = $1;
                        $options{stdout} = 0;
                } elsif (m#^--projectdir=(.+)$#) {
                        $options{projectdir} = $1;
                } else {
                        die "Unknown parameter $_";
                }
        }

        if($options{project}) {
                my $proj_folder = get_project_folder($options{project});
                if(!-d $proj_folder) {
                        die "Project folder `$proj_folder` not found";
                }
        } else {
                die "Unknown project";
        }
}
