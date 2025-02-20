#!/usr/bin/perl

$| = 1;

use strict;
use warnings;
use Data::Dumper;

use lib './perllib';
use Env::Modify;

use File::Basename;
use Cwd 'abs_path';
my $basedirname = dirname(abs_path($0));

use Term::ANSIColor;

our %options = (
        debug => 0,
        project => '',
        int => 0,
        seperator => ';',
        projectdir => './projects',
        maxvalue => undef,
        maxtime => undef,
        keepmongodb => 0,
        nopip => 0,
        dontloadmodules => 0,
        SHOWFAILEDJOBSINPLOT => exists($ENV{SHOWFAILEDJOBSINPLOT}) ? $ENV{SHOWFAILEDJOBSINPLOT} : 0,
        dontchecksqueue => 0
);

use OmniOptFunctions;

analyze_args(@ARGV);

main();

sub main {
        my $projectfolder = get_project_folder($options{project});

        p 1, "Loading modules...";
        modules_load(
		"release/23.04",
		"MongoDB/4.0.3",
		"GCC/11.3.0",
		"OpenMPI/4.1.4",
		"Hyperopt/0.2.7",
		"matplotlib/3.5.2"
        );
        p 10, "Modules loaded";


        modify_system("export SHOWFAILEDJOBSINPLOT=$options{SHOWFAILEDJOBSINPLOT}");
        modify_system('PYTHONPATH=$HOME/.local/lib/python3.7/site-packages:/software/haswell/Python/3.7.4-GCCcore-8.3.0/lib/python3.7/site-packages:$PYTHONPATH');
        if(!$options{nopip}) {
                print "If the module past is missing, install the package future via pip3 install --user future\n";
                print "If the module pymongo is missing, install the package future via pip3 install --user pymongo\n";
        }

        if(!$options{nopip}) {
                myqx("pip3 install --user psutil");
        }

        my $command = "python3 -u script/plot3.py --projectdir=".$options{projectdir}.' --project='.$options{project};
        $command .= " --maxvalue=".$options{maxvalue} if $options{maxvalue};
        $command .= " --maxtime=".$options{maxtime} if $options{maxtime};
        $command .= " --int" if $options{int};
        $command .= " --parameter=".$options{parameter} if $options{parameter};
        $command .= " --switchaxes" if $options{switchaxes};

        p 15, "Getting SLURM-ID";

        my $slurm_id_project = get_id_of_project();

        p 30, "Got Slurm-ID";

        if(exists $ENV{mongodbmachine} && exists $ENV{mongodbport} && length $ENV{mongodbmachine} >= 2 && length $ENV{mongodbport} >= 2) {
                my ($dbip, $dbport) = ($ENV{mongodbmachine}, $ENV{mongodbport});

                warn "Found the following ip:port in \$ENV: $dbip:$dbport\n";

                $command .= " --mongodbmachine=$dbip --mongodbport=$dbport";
                p 40, "Starting Pythonscript (1)";
                print myqx($command, 1);
        } elsif (defined $slurm_id_project) {
                my $ipfilefolder = "$projectfolder/ipfiles";
                if(-d $ipfilefolder) {
                        my ($dbip, $dbport) = map { read_file("$ipfilefolder/$_") } ("mongodbserverip-$slurm_id_project", "mongodbportfile-$slurm_id_project");
                        chomp $dbip;
                        chomp $dbport;

                        warn "Found the following ip:port: $dbip:$dbport\n";
                        $command .= " --mongodbmachine=$dbip --mongodbport=$dbport";
                        p 40, "Starting Pythonscript (2)";
                        print myqx($command);
                } else {
                        die "$ipfilefolder not found";
                }
        } else {
                my $lockfile = "$projectfolder/mongodb/mongod.lock";

                if(!-e $lockfile) {
                        modify_system('export mongodbport=$(bash tools/get_open_port.sh)');
                        print myqx($command);
                        if(!$options{keepmongodb}) {
                                p 99, "Ending MongoDB";
                                myqx("python3 script/endmongodb.py --projectdir=".$options{projectdir}.' --project='.$options{project});
                                p 100, "Ended MongoDB";
                        }
                } else {
                        die "There is a lockfile `$lockfile`, that means the server was either not shut down correctly or it is running in another job. Better not doing anything automatically.\n";
                }
        }
        p 100, "Ended script";
}

sub help {
        my $exit = shift // 0;
        print <<EOF;
tools/plot.pl : Plot database of OmniOpt run

Parameters:

--debug                                 Enables debug mode
--project=NAME                          Sets the name of the project to be plotted
--switchaxes                            Switches X and Y axes in plot
--maxvalue=FLOAT                        Sets a maximal value to get from the DB
--maxtime=INT                           Sets a maximal time (in unix epoch time) to get from the DB
--projectdir=/path/                     Path of the projects
--int                                   Round all non-ints to ints
--keepmongodb                           Do not end MongoDB after plotting
--dontloadmodules                       Disables loading of the modules (e.g. for manual, faster loading)
--nopip                                 Disables psutil installation
--help                                  This help

Needed modules when --dontloadmodules is specified:

ml release/23.04 2>&1 | grep -v load
ml MongoDB/4.0.3 2>&1 | grep -v load
ml GCC/11.3.0 2>&1 | grep -v load
ml OpenMPI/4.1.4 2>&1 | grep -v load
ml Hyperopt/0.2.7 2>&1 | grep -v load
ml matplotlib/3.5.2 2>&1 | grep -v load

EOF
        if($exit) {
                exit 0;
        }
}

sub analyze_args {
        my @args = @_;

        foreach (@args) {
                if(m#^--debug$#) {
                        $options{debug} = 1;
                } elsif (m#^--help$#) {
                        help(1);
                } elsif (m#^--keepmongodb$#) {
                        $options{keepmongodb} = 1;
                } elsif (m#^--dontchecksqueue$#) {
                        $options{dontchecksqueue} = 1;
                } elsif (m#^--nopip$#) {
                        $options{nopip} = 1;
                } elsif (m#^--dontloadmodules$#) {
                        $options{dontloadmodules} = 1;
                } elsif (m#^--project=(.*)$#) {
                        $options{project} = $1;
                } elsif (m#^--switchaxes$#) {
                        $options{switchaxes} = 1;
                } elsif (m#^--maxtime=(\d+)$#) {
                        $options{maxtime} = $1;
                } elsif (m#^--maxvalue=(.+)$#) {
                        $options{maxvalue} = $1;
                } elsif (m#^--parameter=(.+)$#) {
                        $options{parameter} = $1;
                } elsif (m#^--projectdir=(.+)$#) {
                        $options{projectdir} = $1;
                } elsif (m#^--int$#) {
                        $options{int} = 1;
                } else {
                        warn "Unknown parameter $_";
                        help(1);
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
