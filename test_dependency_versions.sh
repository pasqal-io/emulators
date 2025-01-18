mps_string=`grep emu-base ci/emu_mps/pyproject.toml`
[[ "$mps_string" =~ .*([0-9]\.[0-9]\.[0-9]).* ]]
mps_dep=${BASH_REMATCH[1]}
echo mps depends on emu-base version $mps_dep
base_string=`grep version emu_base/__init__.py`
[[ "$base_string" =~ .*([0-9]\.[0-9]\.[0-9]).* ]]
base_version=${BASH_REMATCH[1]}
echo emu-base is version $base_version
if [[ $mps_dep == $base_version ]]
then
exit 0
else
exit 1
fi
