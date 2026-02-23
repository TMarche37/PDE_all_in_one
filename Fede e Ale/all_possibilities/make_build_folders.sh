for dir in */; do
	if [ "$dir" == "common/" ]; then
		echo "Skipping $dir."
	else
		echo "Working on: $dir"
		for subdir in "$dir"*/; do
			if [ -d "$subdir/build" ]; then
				echo "Skipping $subdir, build exists"
			else
				mkdir "$subdir"build
			fi
		done
	fi
done
