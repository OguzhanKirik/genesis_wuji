from setuptools import setup

package_name = 'genesis_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/perception.launch.py']),
        ('share/' + package_name + '/config', ['config/defaults.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'perception_node = genesis_perception.perception_node:main',
            'compare_candidates = genesis_perception.compare_candidates:main',
        ],
    },
)
