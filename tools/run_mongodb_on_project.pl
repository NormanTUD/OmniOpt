#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

use lib './perllib';
use Env::Modify;

our %options = (
        debug => 0,
        project => '',
        int => 0,
        projectdir => './projects',
        query => undef
);

use OmniOptFunctions;

analyze_args(@ARGV);

main();

sub main {
        my $projectfolder = get_project_folder($options{project});

        modules_load(
                "modenv/scs5",
                "MongoDB/4.0.3",
                "Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4"
        );

        if (!$options{project}) {
                die "No --project name was given";
        }

        if (!$options{query}) {
                die "No --query was given";
        }

        my $command = '';

        my $slurm_id_project = get_id_of_project();

        if(defined $slurm_id_project) {
                my $ipfilefolder = "$projectfolder/ipfiles";
                if(-d $ipfilefolder) {
                        my ($dbip, $dbport) = map { read_file("$ipfilefolder/$_") } ("mongodbserverip-$slurm_id_project", "mongodbportfile-$slurm_id_project");
                        chomp $dbip;
                        chomp $dbport;

                        warn "Found the following ip:port: $dbip:$dbport\n";
                        $command = qq#mongo --quiet mongodb://$dbip:$dbport/$options{project} --eval '$options{query}'#;
                        print myqx($command);
                } else {
                        die "$ipfilefolder not found";
                }
        } else {
                warn "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
                warn "! This script needs the database to be running. Start the job and then run this command again !\n";
                warn "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
                warn "\n";
                warn "The slurm job for the project `$options{project}` could not be found. Is it running and has the same slurm name?";

                my $lockfile = "$projectfolder/mongodb/mongod.lock";

                if(!-e $lockfile) {
                        debug "$lockfile DOES NOT exist";
                        print myqx($command);
                        myqx("python3 script/endmongodb.py --projectdir=".$options{projectdir}.' --project='.$options{project});
                } else {
                        debug "$lockfile DOES exist";
                        die "There is a lockfile `$lockfile`, that means the server was either not shut down correctly or it is running in another job. Better not doing anything automatically.\n";
                }
                exit 1;
        }

}

sub analyze_args {
        my @args = @_;

        foreach (@args) {
                if(m#^--debug$#) {
                        $options{debug} = 1;
                } elsif (m#^--project=(.*)$#) {
                        $options{project} = $1;
                } elsif (m#^--switchaxes$#) {
                        $options{switchaxes} = 1;
                } elsif (m#^--query=(.+)$#) {
                        $options{query} = $1;
                } elsif (m#^--parameter=(.+)$#) {
                        $options{parameter} = $1;
                } elsif (m#^--projectdir=(.+)$#) {
                        $options{projectdir} = $1;
                } elsif (m#^--int$#) {
                        $options{int} = 1;
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
