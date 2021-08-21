#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

use lib './perllib';
use Env::Modify;

sub myqx ($);
sub debug ($);

my %options = (
    debug => 0,
    project => '',
    int => 0,
    seperator => ';',
    projectdir => './projects'
);

analyze_args(@ARGV);

main();

sub main {
    my $projectfolder = get_project_folder($options{project});

#'modenv/scs5',
#'MongoDB/4.0.3', 

    #modules_load("MongoDB/4.0.3", "TensorFlow/2.0.0-fosscuda-2019b-Python-3.7.4");
    modules_load("modenv/scs5", "MongoDB/4.0.3", "Hyperopt/0.2.2-fosscuda-2019b-Python-3.7.4");

    modify_system('PYTHONPATH=$HOME/.local/lib/python3.7/site-packages:/software/haswell/Python/3.7.4-GCCcore-8.3.0/lib/python3.7/site-packages:$PYTHONPATH');
    warn "If the module past is missing, install the package future via pip3 install --user future\n";
    warn "If the module pymongo is missing, install the package future via pip3 install --user pymongo\n";

    my $slurm_id_project = get_id_of_project();
    if(defined $slurm_id_project) {
        my $ipfilefolder = "$projectfolder/ipfiles";
        if(-d $ipfilefolder) {
            my ($dbip, $dbport) = map { read_file("$ipfilefolder/$_") } ("mongodbserverip-$slurm_id_project", "mongodbportfile-$slurm_id_project");
            chomp $dbip;
            chomp $dbport;

            warn "Found the following ip:port: $dbip:$dbport\n";
            my $command = qq#python3.7 script/dbtocsv.py #.($options{int} ? '--int=1' : '').qq# --project=$options{project} --mongodbmachine=$dbip --mongodbport=$dbport --seperator="$options{seperator}"#;
            if(-d $options{projectdir}) {
                $command .= " --projectdir=".$options{projectdir}." ";
            }

            print myqx($command);
        } else {
            die "$ipfilefolder not found";
        }
    } else {
        warn "The running slurm job for the project `$options{project}` could not be found. Is it running and has the same slurm name? If not, a new database instance will be started and this warning can be ignored.";

        my $lockfile = "$projectfolder/mongodb/mongod.lock";

        if(!-e $lockfile) {
            my $command = qq#python3.7 script/dbtocsv.py #.($options{int} ? '--int=1' : '').qq# --project=$options{project} --seperator="$options{seperator}"#;
            if(-d $options{projectdir}) {
                $command .= " --projectdir=".$options{projectdir}." ";
            }
            print myqx($command);

            my $command_end_mongodb = qq#python3.7 script/endmongodb.py --project=$options{project} #;
            if($options{projectdir}) {
                $command_end_mongodb .= qq#--projectdir=$options{projectdir}#;
            }

            print myqx($command_end_mongodb);
        } else {
            die "There is a lockfile `$lockfile`, that means the server was either not shut down correctly or it is running in another job. Better not doing anything automatically.\n";
        }
    }
}

sub read_file {
    my $file = shift;

    my $contents = '';
    open my $fh, '<', $file or die "Error opening file $file: $!";

    while (<$fh>) {
        $contents .= $_;
    }

    close $fh;

    return $contents;
}

sub get_id_of_project {
    my %running_jobs = get_running_jobs();

    foreach my $name (keys %running_jobs) {
        debug "$name -> $options{project}?";
        if($name eq $options{project} || $name eq "'$options{project}'") {
            return $running_jobs{$name};
        }
    }

    return undef;
}

sub get_running_jobs {
    my $command = 'sacct --format="JobID,State,JobName%100"';

    my @jobs = map { chomp $_; $_; } myqx $command;

    my %running_jobs = ();

    foreach (@jobs) {
        if(m#^(\d+)\s+RUNNING\s{4,}(.*?)\s*$#) {
            my $id = $1;
            my $name = $2;

            $running_jobs{$name} = $id;
        }
    }

    return %running_jobs;
}

sub myqx ($) {
    my $command = shift;

    debug "command: $command";

    if(wantarray()) {
        my @res = qx($command);
        return @res;
    } else {
        my $res = qx($command);
        return $res;
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

sub get_project_folder {
    my $project = shift;


    return "$options{projectdir}/$project/";
}

sub debug ($) {
    my $arg = shift;
    if($options{debug}) {
        warn "$arg\n";
    }
}

sub modules_load {
    my @modules = @_;
    foreach my $mod (@modules) {
        module_load($mod);
    }

    return 1;
}

sub modify_system {
    my $command = shift;
    debug "modify_system($command)";
    return Env::Modify::system($command);
}

sub module_load {
    my $toload = shift;

    if($toload) {
        my $lmod_path = $ENV{LMOD_CMD};
        my $command = "eval \$($lmod_path sh load $toload)";
        debug $command;
        local $Env::Modify::CMDOPT{startup} = 1;
        modify_system($command);
    } else {
        warn 'Empty module_load!';
    }
    return 1;
}
