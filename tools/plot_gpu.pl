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
        projectdir => './projects',
        logdate => undef,
        dontloadmodules => 0,
        filename => undef
);

use OmniOptFunctions;

analyze_args(@ARGV);

main();

sub main {
        my $projectfolder = get_project_folder($options{project});

        p 1, "Loading modules...";
        modules_load(
                "modenv/scs5",,
                "Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4",
                "matplotlib/3.1.1-foss-2019b-Python-3.7.4"
        );
        p 10, "Modules loaded";


        my $command = qq#python3 script/plot_gpu.py "$options{project}" "$projectfolder/logs/$options{logdate}"#;

        if($options{filename}) {
            $command .= " $options{filename}";
        }

        system($command);

        p 100, "Ending script";
}

sub help {
        my $exit = shift // 0;
        print <<EOF;
tools/plot.pl : Plot GPU-data of an OmniOpt run

Parameters:

--debug                                 Enables debug mode
--project=NAME                          Sets the name of the project to be plotted
--projectdir=/path/                     Path of the projects
--logdate=TIMESTAMPOFLOG                Timestamp of the logfolder to be used
--dontloadmodules                       Disables loading of the modules (e.g. for manual, faster loading)
--help                                  This help

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
                } elsif (m#^--dontloadmodules$#) {
                        $options{dontloadmodules} = 1;
                } elsif (m#^--project=(.*)$#) {
                        $options{project} = $1;
                } elsif (m#^--filename=(.*)$#) {
                        $options{filename} = $1;
                } elsif (m#^--projectdir=(.+)$#) {
                        $options{projectdir} = $1;
                } elsif (m#^--logdate=(.+)$#) {
                        $options{logdate} = $1;
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

        if(!$options{logdate}) {
                die "--logdate must be specified";
        }
}
