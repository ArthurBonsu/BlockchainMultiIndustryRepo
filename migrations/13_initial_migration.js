const NIDRegistry = artifacts.require("NIDRegistry");
const NIASRegistry = artifacts.require("NIASRegistry");
const ABATLTranslation = artifacts.require("ABATLTranslation");
const SequencePathRouter = artifacts.require("SequencePathRouter");
const ClusteringContract = artifacts.require("ClusteringContract");
const fs = require('fs');
const path = require('path');

module.exports = async function(deployer, network, accounts) {
  console.log("Deploying N2N routing system contracts...");
  
  // Deploy NIDRegistry
  console.log("Deploying NIDRegistry...");
  await deployer.deploy(NIDRegistry);
  const nidRegistry = await NIDRegistry.deployed();
  console.log("NIDRegistry deployed at:", nidRegistry.address);
  
  // Deploy NIASRegistry
  console.log("Deploying NIASRegistry...");
  await deployer.deploy(NIASRegistry);
  const niasRegistry = await NIASRegistry.deployed();
  console.log("NIASRegistry deployed at:", niasRegistry.address);
  
  // Deploy ABATLTranslation
  console.log("Deploying ABATLTranslation...");
  await deployer.deploy(ABATLTranslation, nidRegistry.address, niasRegistry.address);
  const abatlTranslation = await ABATLTranslation.deployed();
  console.log("ABATLTranslation deployed at:", abatlTranslation.address);
  
  // Deploy SequencePathRouter
  console.log("Deploying SequencePathRouter...");
  await deployer.deploy(
    SequencePathRouter, 
    nidRegistry.address, 
    niasRegistry.address,
    abatlTranslation.address
  );
  const sequencePathRouter = await SequencePathRouter.deployed();
  console.log("SequencePathRouter deployed at:", sequencePathRouter.address);
  
  // Deploy ClusteringContract
  console.log("Deploying ClusteringContract...");
  await deployer.deploy(
    ClusteringContract,
    nidRegistry.address,
    niasRegistry.address,
    abatlTranslation.address
  );
  const clusteringContract = await ClusteringContract.deployed();
  console.log("ClusteringContract deployed at:", clusteringContract.address);
  
  // Save deployed contract addresses to file
  const addresses = {
    NIDRegistry: nidRegistry.address,
    NIASRegistry: niasRegistry.address,
    ABATLTranslation: abatlTranslation.address,
    SequencePathRouter: sequencePathRouter.address,
    ClusteringContract: clusteringContract.address
  };
  
  const addressesFile = path.join(__dirname, '../contract_addresses.json');
  fs.writeFileSync(addressesFile, JSON.stringify(addresses, null, 2));
  console.log("Contract addresses saved to:", addressesFile);
  
  console.log("All contracts deployed successfully!");
};