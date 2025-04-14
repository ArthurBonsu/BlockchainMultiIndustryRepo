// migrations/12_initial_migration.js
const BCADN = artifacts.require("BCADN");
const ProactiveDefenseMechanism = artifacts.require("ProactiveDefenseMechanism");
const NodeTypes = artifacts.require("NodeTypes");  // If you have this as a separate contract

module.exports = async function(deployer, network, accounts) {
  try {
    // Deploy NodeTypes if it's a separate contract
    // await deployer.deploy(NodeTypes);

    // Deploy BCADN contract
    await deployer.deploy(BCADN);

    // Deploy Proactive Defense Mechanism
    await deployer.deploy(
      ProactiveDefenseMechanism, 
      30,    // anomalyThreshold 
      86400  // baselineUpdateInterval (1 day in seconds)
    );

    // Optional additional contracts you might want to deploy
    // const BCNetworkMonitor = artifacts.require("BCNetworkMonitor");
    // const DynamicNodeWeighting = artifacts.require("DynamicNodeWeighting");
    // const BCShardingManager = artifacts.require("BCShardingManager");
    // const AnomalyDetection = artifacts.require("AnomalyDetection");
    // const ProbabilityGap = artifacts.require("ProbabilityGap");
    // const BCTransactionProcessor = artifacts.require("BCTransactionProcessor");

    console.log(`Deployment completed for network: ${network}`);
  } catch (error) {
    console.error('Deployment error:', error);
    throw error;
  }
};