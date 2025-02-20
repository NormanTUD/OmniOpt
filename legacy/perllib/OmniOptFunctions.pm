package OmniOptFunctions;

use strict;
use warnings;
use Term::ANSIColor;

sub debug ($);
sub p ($;$);

use lib './perllib';
use Env::Modify;

use Exporter qw(import);
 
our @EXPORT = qw(
        get_signal_name
        p
        debug
        myqx
        module_is_loaded
        module_load
        modules_load
        modify_system
        read_file
        get_running_jobs
        get_id_of_project
        module_purge
        get_job_work_dir
        read_ini_file
        get_project_folder
);

sub debug ($) {
        my $arg = shift;
        if($main::options{debug}) {
                warn "$arg\n";
        }
}

sub p ($;$) {
        my $percent = shift;
        my $status = shift // "";
        if($ENV{DISPLAYGAUGE}) {
                warn "PERCENTGAUGE: $percent\n";
                if($status) {
                        warn "GAUGESTATUS: $status\n";
                }
        }
}

sub get_signal_name {
        my $signal = shift;

        if($signal =~ m#^\d+$#) {
                my $name = scalar qx(kill -l $signal);
                chomp $name;
                return $name;
        } else {
                return "unknown";
        }
}

sub myqx ($;$) {
        my $command = shift;
        my $die_on_error = shift // 0;

        debug "command: $command";

        if(wantarray()) {
                my @res = qx($command 2>&1 | grep -v 'DEBUG:matplotlib.font_manager:findfont');
                my $error_code = $?;
                my $exit_code = $error_code >> 8;
                my $signal_code = $error_code & 127;
                warn color("red")."Exited with $exit_code".color("reset")."\n" if $exit_code;
                warn color("red")."Program exited, got signal $signal_code (".get_signal_name($signal_code).")".color("reset")."\n" if $signal_code;
                exit $error_code if($die_on_error && $error_code != 0);
                return @res;
        } else {
                my $res = qx($command 2>&1 | grep -v 'DEBUG:matplotlib.font_manager:findfont');
                my $error_code = $?;
                my $exit_code = $error_code >> 8;
                my $signal_code = $error_code & 127;
                warn color("red")."Exited with $exit_code".color("reset")."\n" if $exit_code;
                warn color("red")."Program exited, got signal $signal_code (".get_signal_name($signal_code).")".color("reset")."\n" if $signal_code;
                exit $error_code if($die_on_error && $error_code != 0);
                return $res;
        }
}

sub module_is_loaded {
        my $module = shift;

        my $lmod_path = $ENV{LMOD_CMD};
        my $command = "eval \$($lmod_path sh is-loaded $module)";
        system($command);
        if($? == 0) {
                return 1;
        }
        return 0;
}

sub module_load {
        my $toload = shift;

        if($main::options{dontloadmodules}) {
                return 1;
        }

        if($toload) {
                if(module_is_loaded($toload)) {
                        warn "$toload already loaded\n";
                } else {
                        my $lmod_path = $ENV{LMOD_CMD};
                        my $command = "eval \$($lmod_path sh load $toload)";
                        debug $command;
                        local $Env::Modify::CMDOPT{startup} = 1;
                        modify_system($command);
                }
        } else {
                warn 'Empty module_load!';
        }

        return 1;
}

sub modules_load {
        my @modules = @_;
        my $skip = 0;

        eval {
                $skip = $main::options{dontloadmodules};
        };

        return if $skip;
        foreach my $mod (@modules) {
                module_load($mod);
        }

        return 1;
}

sub modify_system {
        my $command = shift;
        $command =~ s#\s*$##g;
        debug "modify_system($command)";
        my $values = Env::Modify::system($command);
        my $error_code = $?;

        my $exit_code = $error_code >> 8;
        my $sig_code = $error_code & 127;

        warn "Exit-Code for `$command`: $exit_code" if($exit_code != 0);
        warn "Signal for `$command`: $sig_code" if($sig_code != 0);

        return $values;
}

sub get_project_folder {
        my $project = shift;

        return "$main::options{projectdir}/$project/";
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

sub get_id_of_project {
        if(exists $main::options{dontchecksqueue}) {
                return undef if $main::options{dontchecksqueue} == 1;
        }

        my %running_jobs = get_running_jobs();

        foreach my $name (keys %running_jobs) {
                debug "$name -> $main::options{project}?";
                if($name eq $main::options{project}) {
                        return $running_jobs{$name};
                }
        }

        warn "The slurm job for the project `$main::options{project}` could not be found. Is it running and has the same slurm name? If not, a new database instance will be started and this warning can be ignored.\n";
        return undef;
}

sub module_purge {
        my $lmod_path = $ENV{LMOD_CMD};
        my $command = "eval \$($lmod_path sh purge)";
        local $Env::Modify::CMDOPT{startup} = 1;
        modify_system($command);
}

sub get_job_work_dir {
        my $slurmid = shift;
        my $workdir = '';
        $workdir = qx(scontrol show job $slurmid | grep WorkDir | sed -e 's/.*=//');
        chomp $workdir;
        return $workdir;
}

sub read_ini_file {
        my $file = shift;
        my $conf;
        open (my $INI, $file) || die "Can't open ini-file `$file`: $!\n";
        my $section = '';
        while (<$INI>) {
                chomp;
                if (/^\s*\[\s*(.+?)\s*\]\s*$/) {
                        $section = $1;
                }

                if ( /^\s*([^=]+?)\s*=\s*(.*?)\s*$/ ) {
                        $conf->{$section}->{$1} = $2;
                }
        }
        close ($INI);
        return %{$conf};
}

1;
