"""ManiFlow standard server entry."""

from XPolicyLab.policy.pointcloud_common.model import PointCloudModel


class Model(PointCloudModel):
    policy_name = "ManiFlow"
